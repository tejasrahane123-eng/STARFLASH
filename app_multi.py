import cv2
import mediapipe as mp
import time
import math
import requests

# =========================================================
# SETTINGS
# =========================================================

API_URL = "http://127.0.0.1:5000/api/update"

LOOK_AWAY_LIMIT = 1.0
EYES_CLOSED_LIMIT = 1.5

MAX_STUDENTS = 5
FPS = 30

# =========================================================
# SEND DATA TO FLASK
# =========================================================

def send_data(
    students,
    focused,
    distracted,
    focus_percent,
    total_distractions,
    student_status,
    session_running,
    session_time
):
    data = {
        "students": students,
        "focused": focused,
        "distracted": distracted,
        "focus_percent": round(focus_percent, 1),
        "total_distractions": total_distractions,
        "student_status": student_status,
        "session_running": session_running,
        "session_time": round(session_time, 1),
    }

    try:
        requests.post(API_URL, json=data, timeout=0.5)
    except requests.RequestException:
        pass

# =========================================================
# MEDIAPIPE
# =========================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

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
        vertical_1 = dist(points[eye[1]], points[eye[5]])
        vertical_2 = dist(points[eye[2]], points[eye[4]])
        horizontal = dist(points[eye[0]], points[eye[3]])

        if horizontal == 0:
            return 0.3

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

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

# =========================================================
# MAIN
# =========================================================

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    # -----------------------------------------------------
    # SESSION
    # -----------------------------------------------------

    session_running = False
    session_time = 0.0
    last_session_tick = time.time()

    # -----------------------------------------------------
    # OVERALL METRICS
    # -----------------------------------------------------

    focused_time = 0.0
    distracted_time = 0.0
    total_distractions = 0

    # -----------------------------------------------------
    # PER-STUDENT STATE
    # -----------------------------------------------------

    away_start = {}
    closed_start = {}

    # Prevent repeated counting during one continuous
    # distraction. It becomes False again only after the
    # student returns to a focused state.
    was_distracted = {}

    student_distraction_counts = {}

    # Per-student attention time
    student_focused_time = {}
    student_distracted_time = {}

    last_api_update = 0.0

    print("===================================")
    print(" Student Distraction Detector")
    print("===================================")
    print("S = Start Session")
    print("P = Pause Session")
    print("R = Reset Session")
    print("Q = Quit")
    print("===================================")

    with mp_face_mesh.FaceMesh(
        max_num_faces=MAX_STUDENTS,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read camera.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            current_time = time.time()

            if session_running:
                session_time += current_time - last_session_tick

            last_session_tick = current_time

            students = 0
            distracted_students = 0
            student_status = {}

            # =================================================
            # FACE DETECTION
            # =================================================

            if result.multi_face_landmarks:
                visible_faces = result.multi_face_landmarks[:MAX_STUDENTS]
                students = len(visible_faces)

                for student_id, face_landmarks in enumerate(
                    visible_faces,
                    start=1
                ):
                    h, w, _ = frame.shape

                    points = [
                        (
                            int(landmark.x * w),
                            int(landmark.y * h)
                        )
                        for landmark in face_landmarks.landmark
                    ]

                    direction = get_direction(points)
                    looking_away = direction != "CENTER"

                    left_ratio = eye_ratio(points, LEFT_EYE)
                    right_ratio = eye_ratio(points, RIGHT_EYE)
                    eye_value = (left_ratio + right_ratio) / 2.0
                    eyes_closed = eye_value < 0.18

                    # -----------------------------------------
                    # LOOK-AWAY TIMER
                    # -----------------------------------------

                    if looking_away:
                        away_start.setdefault(student_id, current_time)
                    else:
                        away_start.pop(student_id, None)

                    # -----------------------------------------
                    # CLOSED-EYES TIMER
                    # -----------------------------------------

                    if eyes_closed:
                        closed_start.setdefault(student_id, current_time)
                    else:
                        closed_start.pop(student_id, None)

                    prolonged_away = (
                        student_id in away_start
                        and current_time - away_start[student_id]
                        >= LOOK_AWAY_LIMIT
                    )

                    prolonged_closed = (
                        student_id in closed_start
                        and current_time - closed_start[student_id]
                        >= EYES_CLOSED_LIMIT
                    )

                    is_distracted = prolonged_away or prolonged_closed

                    # -----------------------------------------
                    # INIT STUDENT TIME
                    # -----------------------------------------

                    student_focused_time.setdefault(student_id, 0.0)
                    student_distracted_time.setdefault(student_id, 0.0)
                    student_distraction_counts.setdefault(student_id, 0)
                    was_distracted.setdefault(student_id, False)

                    # -----------------------------------------
                    # ATTENTION
                    # -----------------------------------------

                    if session_running:
                        dt = 1.0 / FPS

                        if is_distracted:
                            student_distracted_time[student_id] += dt
                        else:
                            student_focused_time[student_id] += dt

                    # -----------------------------------------
                    # STATUS + EVENT COUNT
                    # -----------------------------------------

                    if is_distracted:
                        status = "DISTRACTED"
                        distracted_students += 1

                        # Count ONLY the beginning of a new
                        # distraction event.
                        if session_running and not was_distracted[student_id]:
                            total_distractions += 1
                            student_distraction_counts[student_id] += 1

                        was_distracted[student_id] = True

                    elif looking_away:
                        status = "CHECKING..."

                    else:
                        status = "FOCUSED"

                        # Student has recovered, so the next
                        # real distraction will be a new event.
                        was_distracted[student_id] = False

                    total_student_time = (
                        student_focused_time[student_id]
                        + student_distracted_time[student_id]
                    )

                    if total_student_time > 0:
                        attention_percent = (
                            student_focused_time[student_id]
                            / total_student_time
                        ) * 100
                    else:
                        attention_percent = 100.0

                    student_status[str(student_id)] = {
                        "status": status,
                        "direction": direction,
                        "attention": round(attention_percent, 1),
                        "total_distractions": student_distraction_counts[student_id],
                    }

                    # -----------------------------------------
                    # DRAW FACE
                    # -----------------------------------------

                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS
                    )

                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]

                    x_min = max(min(xs) - 10, 0)
                    x_max = min(max(xs) + 10, w - 1)
                    y_min = max(min(ys) - 10, 0)
                    y_max = min(max(ys) + 10, h - 1)

                    cv2.rectangle(
                        frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Student {student_id}",
                        (x_min, max(y_min - 35, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        status,
                        (x_min, max(y_min - 10, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        direction,
                        (x_min, y_max + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        1
                    )

            # =================================================
            # OVERALL FOCUS/DISTRACTION METRICS
            # =================================================

            if session_running and students > 0:
                dt = 1.0 / FPS
                distracted_ratio = distracted_students / students

                distracted_time += dt * distracted_ratio
                focused_time += dt * (1.0 - distracted_ratio)

            total_tracked_time = focused_time + distracted_time

            if total_tracked_time > 0:
                focus_percent = (
                    focused_time / total_tracked_time
                ) * 100.0

                distraction_percent = (
                    distracted_time / total_tracked_time
                ) * 100.0
            else:
                focus_percent = 0.0
                distraction_percent = 0.0

            # =================================================
            # SEND TO DASHBOARD
            # =================================================

            if current_time - last_api_update >= 1.0:
                focused_students = max(
                    students - distracted_students,
                    0
                )

                send_data(
                    students,
                    focused_students,
                    distracted_students,
                    focus_percent,
                    total_distractions,
                    student_status,
                    session_running,
                    session_time
                )

                last_api_update = current_time

            # =================================================
            # CAMERA OVERLAY
            # =================================================

            cv2.rectangle(
                frame,
                (10, 10),
                (390, 260),
                (0, 0, 0),
                -1
            )

            session_text = "RUNNING" if session_running else "STOPPED"

            cv2.putText(
                frame,
                f"SESSION: {session_text}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            hours = int(session_time // 3600)
            minutes = int((session_time % 3600) // 60)
            seconds = int(session_time % 60)

            cv2.putText(
                frame,
                f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}",
                (25, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Students: {students}",
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Distracted Now: {distracted_students}",
                (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Focus: {focus_percent:.1f}%",
                (25, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Distraction: {distraction_percent:.1f}%",
                (25, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Total Distractions: {total_distractions}",
                (25, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "S:Start  P:Pause  R:Reset  Q:Quit",
                (25, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Student Distraction Detector",
                frame
            )

            # =================================================
            # KEYBOARD
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if not session_running:
                    session_running = True
                    last_session_tick = time.time()

                    # Start a fresh event edge.
                    # Any current distraction must persist for
                    # the threshold before it becomes an event.
                    for student_id in list(was_distracted):
                        was_distracted[student_id] = False

            elif key == ord("p"):
                if session_running:
                    now = time.time()
                    session_time += now - last_session_tick
                    last_session_tick = now
                    session_running = False

            elif key == ord("r"):
                session_running = False
                session_time = 0.0
                last_session_tick = time.time()

                focused_time = 0.0
                distracted_time = 0.0
                total_distractions = 0

                away_start.clear()
                closed_start.clear()
                was_distracted.clear()
                student_distraction_counts.clear()

                student_focused_time.clear()
                student_distracted_time.clear()

            elif key == ord("q"):
                send_data(
                    students,
                    max(students - distracted_students, 0),
                    distracted_students,
                    focus_percent,
                    total_distractions,
                    student_status,
                    False,
                    session_time
                )
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
