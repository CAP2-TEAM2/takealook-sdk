import numpy as np
import mediapipe as mp
import time

blink_start_time = time.time()
blink_count = 0
blink_rate = 0

def get_head_pose(landmarks, image_shape):
    image_height, image_width = image_shape
    indices = [33, 263, 1, 152]  # left_eye, right_eye, nose_tip, chin
    points = [
        [landmarks[i].x * image_width, landmarks[i].y * image_height, landmarks[i].z * image_width]
        for i in indices
    ]
    left_eye, right_eye, nose_tip, chin = points
    dx = right_eye[0] - left_eye[0]
    dy = chin[1] - nose_tip[1]
    yaw = np.degrees(np.arctan2(dx, right_eye[2] - left_eye[2]))
    pitch = np.degrees(np.arctan2(dy, chin[2] - nose_tip[2]))
    roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], dx))
    return yaw, pitch, roll

def get_shoulder_roll(landmarks, image_shape):
    image_height, image_width = image_shape
    left = landmarks[11]
    right = landmarks[12]
    dy = (right.y - left.y) * image_height
    dx = (right.x - left.x) * image_width
    roll = np.degrees(np.arctan2(dy, dx))
    return roll

def get_face_distance(landmarks, image_shape):
    image_height, image_width = image_shape
    nose_tip = landmarks[1]
    chin = landmarks[152]
    dy = (chin.y - nose_tip.y) * image_height
    dx = (chin.x - nose_tip.x) * image_width
    distance = np.sqrt(dx**2 + dy**2)
    return distance

def create_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    
def get_blink(landmarks, image_shape):
    global blink_start_time, blink_count, blink_rate
    image_height, image_width = image_shape

    # 왼쪽과 오른쪽 눈의 위쪽/아래쪽 랜드마크
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]

    # 눈 높이 계산
    left_eye_height = (left_eye_bottom.y - left_eye_top.y) * image_height
    right_eye_height = (right_eye_bottom.y - right_eye_top.y) * image_height

    threshold = 1.0  # 눈 감김으로 판단할 기준

    # 눈이 감겼다고 판단되면 깜빡임으로 처리
    if left_eye_height < threshold and right_eye_height < threshold:
        blink_count += 1
        time.sleep(0.1)  # 중복 감지 방지를 위한 짧은 대기

    current_time = time.time()
    elapsed_time = current_time - blink_start_time

    # 5초가 지나면 blink_rate 갱신 후 초기화
    if elapsed_time >= 10.0:
        blink_rate = blink_count
        blink_count = 0
        blink_start_time = current_time

    # 5초 이내엔 현재 카운트 반환, 이후엔 5초 동안 센 횟수 반환
    return blink_count if elapsed_time < 10.0 else blink_rate