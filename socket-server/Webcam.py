import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

shoulder_rolls = []

def get_smoothed_roll(current_roll, window_size=5):
    shoulder_rolls.append(current_roll)
    if len(shoulder_rolls) > window_size:
        shoulder_rolls.pop(0)
    return sum(shoulder_rolls) / len(shoulder_rolls)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        h, w, _ = image.shape
        lx, ly = int(left.x * w), int(left.y * h)
        rx, ry = int(right.x * w), int(right.y * h)

        shoulder_roll = (ry - ly) * 100
        smoothed = get_smoothed_roll(shoulder_roll)

        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
        mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
        mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]

        nx, ny = int(nose.x * w), int(nose.y * h)
        lex, ley = int(left_eye.x * w), int(left_eye.y * h)
        rex, rey = int(right_eye.x * w), int(right_eye.y * h)
        mlx, mly = int(mouth_left.x * w), int(mouth_left.y * h)
        mrx, mry = int(mouth_right.x * w), int(mouth_right.y * h)

        # 얼굴 거리 추정 (눈 사이 거리 기반)
        eye_distance_pixels = abs(rex - lex)
        actual_eye_distance_cm = 6.3  # 평균 눈 사이 거리 (cm)
        focal_length = 600  # 임의 초점 거리 (픽셀 단위, 조정 가능)
        if eye_distance_pixels != 0:
            distance_estimate_cm = (actual_eye_distance_cm * focal_length) / eye_distance_pixels
        else:
            distance_estimate_cm = 0

        # Yaw: 좌우 눈 높이 차이 → 좌우 회전
        yaw = (ley - rey) * 100

        # Pitch: 눈-입 간 수직 거리 변화 → 상하 회전
        eye_avg_y = (ley + rey) / 2
        mouth_avg_y = (mly + mry) / 2
        pitch = (eye_avg_y - mouth_avg_y) * 100

        # Roll: 기존 어깨 기반 기울기 사용
        roll = smoothed

        # 눈과 코 연결
        cv2.line(image, (lex, ley), (nx, ny), (255, 0, 0), 2)
        cv2.line(image, (rex, rey), (nx, ny), (255, 0, 0), 2)

        # 입과 코 연결
        cv2.line(image, (mlx, mly), (nx, ny), (0, 0, 255), 2)
        cv2.line(image, (mrx, mry), (nx, ny), (0, 0, 255), 2)

        # 입 좌우 연결
        cv2.line(image, (mlx, mly), (mrx, mry), (0, 255, 255), 2)

        # 시각화
        cv2.line(image, (lx, ly), (rx, ry), (0, 255, 0), 2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = int(np.mean(gray))

    flipped = cv2.flip(image, 1)
    if results.pose_landmarks:
        cv2.rectangle(flipped, (flipped.shape[1] - 230, 15), (flipped.shape[1] - 10, 55), (0, 0, 0), -1)
        cv2.putText(flipped, f"Yaw: {yaw:.2f}", (flipped.shape[1] - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(flipped, (flipped.shape[1] - 230, 55), (flipped.shape[1] - 10, 95), (0, 0, 0), -1)
        cv2.putText(flipped, f"Pitch: {pitch:.2f}", (flipped.shape[1] - 220, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.rectangle(flipped, (flipped.shape[1] - 230, 85), (flipped.shape[1] - 10, 125), (0, 0, 0), -1)
        cv2.putText(flipped, f"Roll: {roll:.2f}", (flipped.shape[1] - 220, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.rectangle(flipped, (flipped.shape[1] - 230, 115), (flipped.shape[1] - 10, 155), (0, 0, 0), -1)
        cv2.putText(flipped, f"Dist: {distance_estimate_cm:.1f}cm", (flipped.shape[1] - 220, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.rectangle(flipped, (flipped.shape[1] - 230, 145), (flipped.shape[1] - 10, 185), (0, 0, 0), -1)
        cv2.putText(flipped, f"Shoulder: {smoothed:.2f}", (flipped.shape[1] - 220, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (144, 238, 144), 2)
        cv2.rectangle(flipped, (flipped.shape[1] - 230, 175), (flipped.shape[1] - 10, 215), (0, 0, 0), -1)
        cv2.putText(flipped, f"Brightness: {brightness}", (flipped.shape[1] - 220, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Shoulder Roll", flipped)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()