import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
import mediapipe as mp
import math

# Mediapipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 각도 계산 함수
def calculate_angle(a, b, c):
    def get_angle(p1, p2):
        return math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    angle1 = get_angle(a, b)
    angle2 = get_angle(c, b)
    return abs(math.degrees(angle1 - angle2))

# 웹캡처 초기화
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 좌우 반전
        image = cv2.flip(image, 1)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 좌우 반전을 고려한 어깨 좌표 변경
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            # 좌우 반전된 이미지에서 좌표 반전
            left_shoulder[0] = 1 - left_shoulder[0]  # x 좌표 반전
            right_shoulder[0] = 1 - right_shoulder[0]  # x 좌표 반전

            # 다른 랜드마크 및 각도 계산
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2,
                            (left_shoulder[1] + right_shoulder[1]) / 2]

            # 거북목(고개 숙임) 각도 측정
            neck_angle = calculate_angle(left_shoulder, mid_shoulder, nose)

            # 어깨 기울기 각도 측정
            shoulder_angle = calculate_angle(
                [left_shoulder[0], left_shoulder[1] - 0.1],
                left_shoulder,
                right_shoulder
            )

            # 얼굴 거리 감지
            face_distance = abs(landmarks[mp_pose.PoseLandmark.NOSE.value].z)

            # 어깨 좌표 표시 (좌측 어깨는 녹색, 우측 어깨는 빨간색)
            cv2.circle(image, (int(left_shoulder[0] * image.shape[1]), int(left_shoulder[1] * image.shape[0])), 5, (0, 255, 0), -1)
            cv2.circle(image, (int(right_shoulder[0] * image.shape[1]), int(right_shoulder[1] * image.shape[0])), 5, (0, 0, 255), -1)
            cv2.putText(image, 'Left Shoulder', (int(left_shoulder[0] * image.shape[1]) + 10, int(left_shoulder[1] * image.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, 'Right Shoulder', (int(right_shoulder[0] * image.shape[1]) + 10, int(right_shoulder[1] * image.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 디버그 텍스트 표시
            cv2.putText(image, f'Neck Angle: {int(neck_angle)}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f'Distance: {round(face_distance, 2)}', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(image, f'Shoulder Angle: {int(shoulder_angle)}', (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Posture Monitor', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
            break

cap.release()
cv2.destroyAllWindows()
