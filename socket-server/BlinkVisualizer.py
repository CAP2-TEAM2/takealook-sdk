import cv2
import numpy as np
import time
import PoseEstimation as pe  # 같은 폴더 내에 PoseEstimation.py가 있어야 함

# 얼굴 메쉬 모델 초기화
face_mesh = pe.create_face_mesh()

# 웹캠 켜기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 인식은 RGB 이미지에서 수행
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    status_text = "No Face"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        blink_rate = pe.get_blink(landmarks, frame.shape[:2])

        # 현재 눈 상태 판단 (시각화용)
        left_eye_height = (landmarks[145].y - landmarks[159].y) * frame.shape[0]
        right_eye_height = (landmarks[374].y - landmarks[386].y) * frame.shape[0]
        threshold = 1.0

        if left_eye_height < threshold and right_eye_height < threshold:
            status_text = f"Eyes Closed (Blinks/s: {blink_rate})"
            color = (0, 0, 255)
        else:
            status_text = f"Eyes Open (Blinks/s: {blink_rate})"
            color = (0, 255, 0)

    # 화면에 상태 출력
    cv2.putText(frame, status_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Blink Visualizer", frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()