from flask import Flask, jsonify, request, send_file, Response, send_from_directory
import cv2
import mediapipe as mp
import time
import math
import threading
import os
import uuid
import json
from datetime import datetime
import numpy as np

app = Flask(__name__)

# =========================================================
# GLOBAL DASHBOARD STATE
# =========================================================

ai_data = {
    "students": 0,
    "focused": 0,
    "distracted": 0,
    "focus_percent": 0.0,
    "total_distractions": 0,
    "student_status": {},
    "session_running": False,
    "session_time": 0.0,
    "camera_ok": False,
    "source_mode": "local",
    "remote_students": 0,
}

state_lock = threading.Lock()
frame_lock = threading.Lock()
latest_frame = None
camera_started = False

session_running = False
session_time = 0.0
last_session_tick = time.time()
focused_time = 0.0
distracted_time = 0.0
total_distractions = 0

# Persistent Storage Room
STORAGE_ROOM_FILE = os.path.join(os.path.dirname(__file__), "storage_room.json")
storage_lock = threading.Lock()
current_session_id = None
current_session_started_at = None

# Local-camera per-student state
away_start = {}
closed_start = {}
was_distracted = {}
student_distraction_counts = {}
student_focused_time = {}
student_distracted_time = {}

LOOK_AWAY_LIMIT = 1.0
EYES_CLOSED_LIMIT = 1.5
MAX_STUDENTS = 5
FPS = 30

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# =========================================================
# MULTI-REMOTE STATE
# =========================================================

remote_lock = threading.Lock()
remote_detector_lock = threading.Lock()
REMOTE_TIMEOUT = 8.0
MAX_REMOTE_STUDENTS = 8

# participant_id -> participant state
remote_participants = {}

remote_face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =========================================================
# STORAGE ROOM
# =========================================================

def storage_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_storage_reports():
    try:
        if not os.path.exists(STORAGE_ROOM_FILE):
            return []
        with open(STORAGE_ROOM_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            reports = data.get("reports", [])
            return reports if isinstance(reports, list) else []
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return []


def save_storage_reports(reports):
    folder = os.path.dirname(STORAGE_ROOM_FILE)
    os.makedirs(folder, exist_ok=True)
    temp_file = STORAGE_ROOM_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump({"reports": reports}, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, STORAGE_ROOM_FILE)


def append_storage_report(report):
    with storage_lock:
        reports = load_storage_reports()
        reports.append(report)
        save_storage_reports(reports)


def build_remote_student_report(participant_id, info, end_reason):
    focused = float(info.get("focused_time", 0.0))
    distracted = float(info.get("distracted_time", 0.0))
    tracked = focused + distracted
    attention = (focused / tracked * 100.0) if tracked > 0 else 100.0

    direction_counts = info.get("direction_counts", {}) or {}
    direction_counts = {
        "CENTER": int(direction_counts.get("CENTER", 0)),
        "LEFT": int(direction_counts.get("LEFT", 0)),
        "RIGHT": int(direction_counts.get("RIGHT", 0)),
    }

    return {
        "report_type": "student",
        "session_id": info.get("session_id") or current_session_id or "unsaved-session",
        "student_id": participant_id,
        "student_name": info.get("name") or f"Student {participant_id[-4:]}",
        "joined_at": info.get("connected_at"),
        "session_started_at": current_session_started_at,
        "left_at": storage_now(),
        "saved_at": storage_now(),
        "end_reason": end_reason,
        "duration_seconds": round(tracked, 1),
        "focused_time_seconds": round(focused, 1),
        "distracted_time_seconds": round(distracted, 1),
        "attention_percent": round(attention, 1),
        "distraction_percent": round((distracted / tracked * 100.0) if tracked > 0 else 0.0, 1),
        "total_distractions": int(info.get("distraction_count", 0)),
        "last_status": info.get("status", "WAITING"),
        "last_direction": info.get("direction", "CENTER"),
        "direction_counts": direction_counts,
        "source": "remote",
    }


def save_remote_student_report(participant_id, info, end_reason):
    if info.get("report_saved"):
        return False

    # Save only meaningful session data. A student who merely opened the
    # join page before the session started does not create an empty report.
    tracked = float(info.get("focused_time", 0.0)) + float(info.get("distracted_time", 0.0))
    if tracked <= 0 and not info.get("session_id"):
        return False

    report = build_remote_student_report(participant_id, info, end_reason)
    append_storage_report(report)
    info["report_saved"] = True
    return True


def save_pending_remote_reports(end_reason):
    saved = 0
    with remote_lock:
        pending = [(pid, info) for pid, info in remote_participants.items() if not info.get("report_saved")]
        for pid, info in pending:
            if save_remote_student_report(pid, info, end_reason):
                saved += 1
    return saved


def create_new_session_id():
    return "session-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


# =========================================================
# HELPERS
# =========================================================

def dist(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def eye_ratio(points, eye):
    try:
        v1 = dist(points[eye[1]], points[eye[5]])
        v2 = dist(points[eye[2]], points[eye[4]])
        h = dist(points[eye[0]], points[eye[3]])
        return 0.3 if h == 0 else (v1 + v2) / (2.0 * h)
    except (IndexError, TypeError, ZeroDivisionError):
        return 0.3


def get_direction(points):
    try:
        nose = points[1]
        left_eye = points[33]
        right_eye = points[263]

        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_width = abs(right_eye[0] - left_eye[0])

        if eye_width == 0:
            return "CENTER"

        ratio = (nose[0] - eye_center_x) / eye_width

        if ratio > 0.20:
            return "RIGHT"
        if ratio < -0.20:
            return "LEFT"
        return "CENTER"

    except (IndexError, TypeError, ZeroDivisionError):
        return "CENTER"


def format_time(seconds):
    total = max(0, int(seconds))
    return (
        f"{total // 3600:02d}:"
        f"{(total % 3600) // 60:02d}:"
        f"{total % 60:02d}"
    )


def make_participant_id():
    return "stu-" + uuid.uuid4().hex[:10]


def active_remote_participants():
    now = time.time()
    stale_reports = []
    with remote_lock:
        stale = [
            pid
            for pid, info in remote_participants.items()
            if now - info["last_seen"] > REMOTE_TIMEOUT
        ]
        for pid in stale:
            info = remote_participants.pop(pid, None)
            if info is not None and not info.get("report_saved"):
                stale_reports.append((pid, info))
        active = dict(remote_participants)

    for pid, info in stale_reports:
        try:
            save_remote_student_report(pid, info, "connection_timeout")
        except OSError:
            pass

    return active


def remote_count():
    return len(active_remote_participants())


def reset_remote_metrics(clear_session_identity=False):
    with remote_lock:
        for info in remote_participants.values():
            info["away_start"] = None
            info["closed_start"] = None
            info["was_distracted"] = False
            info["distraction_count"] = 0
            info["focused_time"] = 0.0
            info["distracted_time"] = 0.0
            info["status"] = "WAITING"
            info["direction"] = "CENTER"
            info["attention"] = 100.0
            info["direction_counts"] = {"CENTER": 0, "LEFT": 0, "RIGHT": 0}
            info["report_saved"] = False
            if clear_session_identity:
                info["session_id"] = None


def refresh_remote_ai_data():
    """Aggregate currently active remote participants into dashboard state."""
    global focused_time, distracted_time

    participants = active_remote_participants()

    with state_lock:
        running = session_running
        current_time = session_time
        total_dis = total_distractions

    status_map = {}
    focused_count = 0
    distracted_count = 0
    total_focused = 0.0
    total_distracted = 0.0

    for participant_id, info in participants.items():
        status = info.get("status", "WAITING")
        if status == "FOCUSED":
            focused_count += 1
        elif status == "DISTRACTED":
            distracted_count += 1

        total_focused += float(info.get("focused_time", 0.0))
        total_distracted += float(info.get("distracted_time", 0.0))

        status_map[participant_id] = {
            "name": info.get("name") or f"Student {participant_id}",
            "status": status,
            "direction": info.get("direction", "CENTER"),
            "attention": round(float(info.get("attention", 100.0)), 1),
            "total_distractions": int(info.get("distraction_count", 0)),
        }

    total_tracked = total_focused + total_distracted
    focus_percent = (
        (total_focused / total_tracked) * 100.0
        if total_tracked > 0 else 0.0
    )

    if participants:
        ai_data.update({
            "students": len(participants),
            "focused": focused_count,
            "distracted": distracted_count,
            "focus_percent": round(focus_percent, 1),
            "total_distractions": total_dis,
            "student_status": status_map,
            "session_running": running,
            "session_time": round(current_time, 1),
            "camera_ok": True,
            "source_mode": "remote",
            "remote_students": len(participants),
        })


def start_session():
    global session_running, last_session_tick
    global current_session_id, current_session_started_at

    with state_lock:
        if session_running:
            return

        current_session_id = create_new_session_id()
        current_session_started_at = storage_now()
        session_running = True
        last_session_tick = time.time()

    with remote_lock:
        for info in remote_participants.values():
            info["session_id"] = current_session_id
            info["report_saved"] = False
            info["away_start"] = None
            info["closed_start"] = None
            info["was_distracted"] = False
            info["distraction_count"] = 0
            info["focused_time"] = 0.0
            info["distracted_time"] = 0.0
            info["direction_counts"] = {"CENTER": 0, "LEFT": 0, "RIGHT": 0}
            info["status"] = "WAITING"
            info["direction"] = "CENTER"
            info["attention"] = 100.0


def pause_session():
    global session_running, session_time, last_session_tick
    with state_lock:
        if session_running:
            now = time.time()
            session_time += now - last_session_tick
            last_session_tick = now
            session_running = False


def end_session():
    global session_running, session_time, last_session_tick

    with state_lock:
        now = time.time()
        if session_running:
            session_time += now - last_session_tick
        last_session_tick = now
        session_running = False

    saved = save_pending_remote_reports("session_ended")
    return saved


def reset_session():
    global session_running, session_time, last_session_tick
    global focused_time, distracted_time, total_distractions
    global current_session_id, current_session_started_at

    # Preserve the current session in Storage Room before clearing it.
    save_pending_remote_reports("reset")

    with state_lock:
        session_running = False
        session_time = 0.0
        last_session_tick = time.time()
        focused_time = 0.0
        distracted_time = 0.0
        total_distractions = 0
        current_session_id = None
        current_session_started_at = None

        away_start.clear()
        closed_start.clear()
        was_distracted.clear()
        student_distraction_counts.clear()
        student_focused_time.clear()
        student_distracted_time.clear()

        ai_data.update({
            "students": 0,
            "focused": 0,
            "distracted": 0,
            "focus_percent": 0.0,
            "total_distractions": 0,
            "student_status": {},
            "session_running": False,
            "session_time": 0.0,
            "source_mode": "local",
            "remote_students": 0,
        })

    reset_remote_metrics(clear_session_identity=True)


# =========================================================
# REMOTE FRAME PROCESSING
# =========================================================

def process_remote_frame(participant_id, frame):
    global session_time, last_session_tick, total_distractions

    now = time.time()

    with remote_detector_lock:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = remote_face_mesh.process(rgb)

    with remote_lock:
        info = remote_participants.get(participant_id)
        if info is None:
            return False
        info["last_seen"] = now

    status = "NO FACE"
    direction = "CENTER"
    attention = 100.0
    is_distracted = False

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        h, w, _ = frame.shape

        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in face_landmarks.landmark
        ]

        direction = get_direction(points)
        looking_away = direction != "CENTER"

        eye_value = (
            eye_ratio(points, LEFT_EYE) +
            eye_ratio(points, RIGHT_EYE)
        ) / 2.0
        eyes_closed = eye_value < 0.18

        with remote_lock:
            info = remote_participants.get(participant_id)
            if info is None:
                return False

            direction_counts = info.setdefault(
                "direction_counts",
                {"CENTER": 0, "LEFT": 0, "RIGHT": 0}
            )
            direction_counts[direction] = int(direction_counts.get(direction, 0)) + 1

            if looking_away:
                if info["away_start"] is None:
                    info["away_start"] = now
            else:
                info["away_start"] = None

            if eyes_closed:
                if info["closed_start"] is None:
                    info["closed_start"] = now
            else:
                info["closed_start"] = None

            prolonged_away = (
                info["away_start"] is not None and
                now - info["away_start"] >= LOOK_AWAY_LIMIT
            )

            prolonged_closed = (
                info["closed_start"] is not None and
                now - info["closed_start"] >= EYES_CLOSED_LIMIT
            )

            is_distracted = prolonged_away or prolonged_closed

            with state_lock:
                running = session_running

            if running:
                dt = 0.25
                if is_distracted:
                    info["distracted_time"] += dt
                else:
                    info["focused_time"] += dt

            if is_distracted:
                status = "DISTRACTED"
                if running and not info["was_distracted"]:
                    total_distractions += 1
                    info["distraction_count"] += 1
                info["was_distracted"] = True
            elif looking_away:
                status = "CHECKING..."
            else:
                status = "FOCUSED"
                info["was_distracted"] = False

            total_student_time = info["focused_time"] + info["distracted_time"]
            attention = (
                info["focused_time"] / total_student_time * 100.0
                if total_student_time > 0 else 100.0
            )

            info["status"] = status
            info["direction"] = direction
            info["attention"] = round(attention, 1)

        # Draw face overlay onto the participant's frame.
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
        )

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min = max(min(xs) - 10, 0)
        x_max = min(max(xs) + 10, w - 1)
        y_min = max(min(ys) - 10, 0)
        y_max = min(max(ys) + 10, h - 1)

        box_color = (220, 38, 38) if is_distracted else (22, 163, 74)

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            box_color,
            2,
        )

        with remote_lock:
            display_name = info.get("name") or f"Student {participant_id}"

        cv2.putText(
            frame,
            display_name,
            (x_min, max(y_min - 30, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            status,
            (x_min, max(y_min - 8, 45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            direction,
            (x_min, min(y_max + 22, h - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )

    else:
        with remote_lock:
            info = remote_participants.get(participant_id)
            if info is None:
                return False
            info["status"] = "NO FACE"
            info["direction"] = "CENTER"
            info["attention"] = round(
                info["focused_time"] /
                (info["focused_time"] + info["distracted_time"]) * 100.0,
                1,
            ) if (info["focused_time"] + info["distracted_time"]) > 0 else 100.0

        cv2.putText(
            frame,
            "No face detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

    # Advance the single global session clock once per received frame.
    with state_lock:
        if session_running:
            session_time += now - last_session_tick
            last_session_tick = now
        else:
            last_session_tick = now

    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )

    if ok:
        with remote_lock:
            info = remote_participants.get(participant_id)
            if info is not None:
                info["frame"] = buffer.tobytes()
                info["last_seen"] = now

    refresh_remote_ai_data()
    return True


# =========================================================
# LOCAL CAMERA WORKER
# =========================================================

def camera_worker():
    global latest_frame, session_time, last_session_tick
    global focused_time, distracted_time, total_distractions

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        with state_lock:
            ai_data["camera_ok"] = False
        return

    with state_lock:
        ai_data["camera_ok"] = True

    with mp_face_mesh.FaceMesh(
        max_num_faces=MAX_STUDENTS,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            now = time.time()
            remotes = active_remote_participants()
            has_remote = bool(remotes)

            # Local source only advances the session clock while no remote
            # students are connected. Remote uploads own the clock otherwise.
            with state_lock:
                running = session_running
                if running and not has_remote:
                    session_time += now - last_session_tick
                    last_session_tick = now
                elif not running:
                    last_session_tick = now

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            if has_remote:
                # Remote students are shown through /video_feed mosaic.
                # The local camera remains available but does not overwrite
                # remote dashboard data while remote participants are active.
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            students = 0
            distracted_students = 0
            student_status = {}

            if result.multi_face_landmarks:
                visible = result.multi_face_landmarks[:MAX_STUDENTS]
                students = len(visible)
                h, w, _ = frame.shape

                for student_id, face_landmarks in enumerate(visible, start=1):
                    points = [
                        (int(lm.x * w), int(lm.y * h))
                        for lm in face_landmarks.landmark
                    ]

                    direction = get_direction(points)
                    looking_away = direction != "CENTER"
                    eye_value = (
                        eye_ratio(points, LEFT_EYE) +
                        eye_ratio(points, RIGHT_EYE)
                    ) / 2.0
                    eyes_closed = eye_value < 0.18

                    if looking_away:
                        away_start.setdefault(student_id, now)
                    else:
                        away_start.pop(student_id, None)

                    if eyes_closed:
                        closed_start.setdefault(student_id, now)
                    else:
                        closed_start.pop(student_id, None)

                    prolonged_away = (
                        student_id in away_start and
                        now - away_start[student_id] >= LOOK_AWAY_LIMIT
                    )
                    prolonged_closed = (
                        student_id in closed_start and
                        now - closed_start[student_id] >= EYES_CLOSED_LIMIT
                    )
                    is_distracted = prolonged_away or prolonged_closed

                    student_focused_time.setdefault(student_id, 0.0)
                    student_distracted_time.setdefault(student_id, 0.0)
                    student_distraction_counts.setdefault(student_id, 0)
                    was_distracted.setdefault(student_id, False)

                    if running:
                        dt = 1.0 / FPS
                        if is_distracted:
                            student_distracted_time[student_id] += dt
                        else:
                            student_focused_time[student_id] += dt

                    if is_distracted:
                        status = "DISTRACTED"
                        distracted_students += 1
                        if running and not was_distracted[student_id]:
                            total_distractions += 1
                            student_distraction_counts[student_id] += 1
                        was_distracted[student_id] = True
                    elif looking_away:
                        status = "CHECKING..."
                    else:
                        status = "FOCUSED"
                        was_distracted[student_id] = False

                    total_student_time = (
                        student_focused_time[student_id] +
                        student_distracted_time[student_id]
                    )
                    attention = (
                        student_focused_time[student_id] /
                        total_student_time * 100.0
                        if total_student_time > 0 else 100.0
                    )

                    student_status[str(student_id)] = {
                        "name": f"Student {student_id}",
                        "status": status,
                        "direction": direction,
                        "attention": round(attention, 1),
                        "total_distractions": student_distraction_counts[student_id],
                    }

                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                    )

                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x_min = max(min(xs) - 10, 0)
                    x_max = min(max(xs) + 10, w - 1)
                    y_min = max(min(ys) - 10, 0)
                    y_max = min(max(ys) + 10, h - 1)
                    box_color = (220, 38, 38) if is_distracted else (22, 163, 74)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                    cv2.putText(frame, f"Student {student_id}", (x_min, max(y_min - 30, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(frame, status, (x_min, max(y_min - 8, 45)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(frame, direction, (x_min, min(y_max + 22, h - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            if running and students > 0:
                dt = 1.0 / FPS
                ratio = distracted_students / students
                distracted_time += dt * ratio
                focused_time += dt * (1.0 - ratio)

            total_tracked = focused_time + distracted_time
            focus_percent = (
                focused_time / total_tracked * 100.0
                if total_tracked > 0 else 0.0
            )
            focused_students = max(students - distracted_students, 0)

            with state_lock:
                ai_data.update({
                    "students": students,
                    "focused": focused_students,
                    "distracted": distracted_students,
                    "focus_percent": round(focus_percent, 1),
                    "total_distractions": total_distractions,
                    "student_status": student_status,
                    "session_running": session_running,
                    "session_time": round(session_time, 1),
                    "camera_ok": True,
                    "source_mode": "local",
                    "remote_students": 0,
                })

            session_text = "RUNNING" if running else "PAUSED"
            cv2.rectangle(frame, (12, 12), (325, 104), (17, 24, 39), -1)
            cv2.putText(frame, f"SESSION: {session_text}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"TIME: {format_time(session_time)}", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            cv2.putText(frame, f"FOCUS: {focus_percent:.1f}%", (25, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82],
            )
            if ok:
                with frame_lock:
                    latest_frame = buffer.tobytes()

            time.sleep(0.001)


def ensure_camera_thread():
    global camera_started
    if camera_started:
        return
    camera_started = True
    threading.Thread(target=camera_worker, daemon=True).start()


# =========================================================
# ROUTES
# =========================================================
@app.route("/project_video")
def project_video():
    video_path = os.path.join(
        os.path.dirname(__file__),
        "static",
        "videos",
        "starflash_about.mp4"
    )

    return send_file(
        video_path,
        mimetype="video/mp4",
        conditional=True
    )
@app.route("/")
def dashboard():
    return send_file(
        os.path.join(os.path.dirname(__file__), "templates", "index_chart.html")
    )


@app.route("/manifest.json")
def manifest():
    return send_file(
        os.path.join(os.path.dirname(__file__), "templates", "manifest.json"),
        mimetype="application/manifest+json",
    )


@app.route("/join")
def remote_join():
    return send_file(
        os.path.join(os.path.dirname(__file__), "templates", "join.html")
    )


@app.route("/api/data")
def get_data():
    # Keep the dashboard data synchronized with remote participants.
    if remote_count() > 0:
        refresh_remote_ai_data()
    with state_lock:
        return jsonify(dict(ai_data))


@app.route("/api/session", methods=["POST"])
def session_control():
    data = request.get_json(silent=True) or {}
    action = str(data.get("action", "")).lower()

    saved_reports = 0

    if action == "start":
        start_session()
    elif action == "pause":
        pause_session()
    elif action == "end":
        saved_reports = end_session()
    elif action == "reset":
        reset_session()
    else:
        return jsonify({"success": False, "error": "Unknown action"}), 400

    with state_lock:
        return jsonify({
            "success": True,
            "session_running": session_running,
            "session_time": round(session_time, 1),
            "saved_reports": saved_reports,
        })


@app.route("/api/storage-room")
def storage_room_api():
    with storage_lock:
        reports = load_storage_reports()
    reports.sort(key=lambda item: item.get("saved_at", ""), reverse=True)
    return jsonify({
        "success": True,
        "count": len(reports),
        "reports": reports,
    })


@app.route("/api/storage-room/clear", methods=["POST"])
def storage_room_clear():
    with storage_lock:
        save_storage_reports([])
    return jsonify({"success": True})


@app.route("/api/remote/status")
def remote_status():
    participants = active_remote_participants()
    return jsonify({
        "active": bool(participants),
        "count": len(participants),
        "source_mode": "remote" if participants else "local",
        "students": [
            {
                "participant_id": pid,
                "name": info.get("name") or f"Student {pid}",
                "status": info.get("status", "WAITING"),
            }
            for pid, info in participants.items()
        ],
    })


@app.route("/api/remote/start", methods=["POST"])
def remote_start():
    data = request.get_json(silent=True) or {}
    participant_id = str(data.get("participant_id", "")).strip()
    name = str(data.get("name", "")).strip()[:40]

    if not participant_id:
        participant_id = make_participant_id()

    if not name:
        name = f"Student {participant_id[-4:]}"

    now = time.time()

    with remote_lock:
        existing = remote_participants.get(participant_id)

        active_now = sum(
            1
            for info in remote_participants.values()
            if now - info["last_seen"] <= REMOTE_TIMEOUT
        )

        if existing is None and active_now >= MAX_REMOTE_STUDENTS:
            return jsonify({
                "success": False,
                "error": f"Maximum {MAX_REMOTE_STUDENTS} remote students are currently connected.",
            }), 409

        if existing is None:
            with state_lock:
                active_session_id = current_session_id if session_running else None

            remote_participants[participant_id] = {
                "name": name,
                "last_seen": now,
                "connected_at": storage_now(),
                "frame": None,
                "status": "WAITING",
                "direction": "CENTER",
                "attention": 100.0,
                "distraction_count": 0,
                "focused_time": 0.0,
                "distracted_time": 0.0,
                "away_start": None,
                "closed_start": None,
                "was_distracted": False,
                "direction_counts": {"CENTER": 0, "LEFT": 0, "RIGHT": 0},
                "session_id": active_session_id,
                "report_saved": False,
            }
        else:
            existing["name"] = name
            existing["last_seen"] = now

    with state_lock:
        ai_data["source_mode"] = "remote"
        ai_data["camera_ok"] = True
        ai_data["remote_students"] = len(active_remote_participants())

    return jsonify({
        "success": True,
        "participant_id": participant_id,
        "name": name,
    })


@app.route("/api/remote/frame", methods=["POST"])
def remote_frame_upload():
    participant_id = str(request.form.get("participant_id", "")).strip()
    if not participant_id:
        return jsonify({"success": False, "error": "Missing participant_id."}), 400

    file = request.files.get("frame")
    if file is None:
        return jsonify({"success": False, "error": "No frame received."}), 400

    raw = file.read()
    if not raw:
        return jsonify({"success": False, "error": "Empty frame."}), 400

    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        return jsonify({"success": False, "error": "Invalid image."}), 400

    with remote_lock:
        info = remote_participants.get(participant_id)
        if info is None:
            return jsonify({
                "success": False,
                "error": "Participant session expired. Please reconnect.",
            }), 409
        info["last_seen"] = time.time()

    ok = process_remote_frame(participant_id, image)
    if not ok:
        return jsonify({"success": False, "error": "Participant session expired."}), 409

    with state_lock:
        return jsonify({
            "success": True,
            "students": ai_data.get("students", 0),
            "focus_percent": ai_data.get("focus_percent", 0.0),
            "student_status": ai_data.get("student_status", {}),
        })


@app.route("/api/remote/stop", methods=["POST"])
def remote_stop():
    participant_id = str(
        (request.get_json(silent=True) or {}).get("participant_id", "")
    ).strip()

    if participant_id:
        with remote_lock:
            info = remote_participants.get(participant_id)

        if info is not None:
            try:
                save_remote_student_report(participant_id, info, "student_left")
            except OSError:
                pass

        with remote_lock:
            remote_participants.pop(participant_id, None)
    else:
        # Backward-compatible: only clear nothing when an ID is omitted.
        pass

    if remote_count() == 0:
        with state_lock:
            ai_data["source_mode"] = "local"
            ai_data["remote_students"] = 0

    else:
        refresh_remote_ai_data()

    return jsonify({"success": True})


def build_remote_mosaic():
    participants = active_remote_participants()
    frames = []

    for pid, info in participants.items():
        frame_bytes = info.get("frame")
        if not frame_bytes:
            continue
        image = cv2.imdecode(
            np.frombuffer(frame_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is not None:
            frames.append(image)

    if not frames:
        return None

    tile_w = 420
    tile_h = 315
    cols = 2
    rows = math.ceil(len(frames) / cols)

    canvas = np.zeros(
        (rows * tile_h, cols * tile_w, 3),
        dtype=np.uint8,
    )
    canvas[:] = (8, 15, 28)

    for index, frame in enumerate(frames):
        resized = cv2.resize(frame, (tile_w, tile_h))
        row = index // cols
        col = index % cols
        y1 = row * tile_h
        x1 = col * tile_w
        canvas[y1:y1 + tile_h, x1:x1 + tile_w] = resized

    count = len(frames)
    cv2.rectangle(canvas, (12, 12), (260, 55), (17, 24, 39), -1)
    cv2.putText(
        canvas,
        f"REMOTE STUDENTS: {count}",
        (24, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (96, 165, 250),
        2,
    )

    return canvas


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            mosaic = build_remote_mosaic()

            if mosaic is not None:
                ok, buffer = cv2.imencode(
                    ".jpg",
                    mosaic,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )
                frame = buffer.tobytes() if ok else None
            else:
                with frame_lock:
                    frame = latest_frame

            if frame is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    ensure_camera_thread()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
