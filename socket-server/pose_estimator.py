import numpy as np
import mediapipe as mp

PITCH_OFFSET = -20
MIN_BRIGHTNESS = 150
CLOSE_DISTANCE = 30
SHOULDER_DEGREE = 5
LOOK_DOWN = 28

def get_code(yaw, pitch, roll, distance, shoulder_roll, shoulder_dis, brightness):

    # too dark / too close / turtle neck / shoulder
    result = [0, 0, 0, 0]

    result[0] = int(brightness <= MIN_BRIGHTNESS) + 1
    result[1] = int(distance > CLOSE_DISTANCE) + 1
    result[2] = int(shoulder_dis > 0) + 1
    result[3] = int(abs(shoulder_roll - 90) > SHOULDER_DEGREE) + 1

    return result

def get_head_pose(landmarks, image_shape):
    image_height, image_width = image_shape
    indices = [33, 263, 1, 152]
    points = [
        [landmarks[i].x * image_width, landmarks[i].y * image_height, landmarks[i].z * image_width]
        for i in indices
    ]
    left_eye, right_eye, nose_tip, chin = points

    dx = right_eye[0] - left_eye[0]
    dy = chin[1] - nose_tip[1]

    yaw = np.degrees(np.arctan2(dx, right_eye[2] - left_eye[2]))
    raw_pitch = np.degrees(np.arctan2(dy, chin[2] - nose_tip[2]))
    pitch = raw_pitch + PITCH_OFFSET

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

def get_shoulder_distance(landmarks, image_shape):
    left = landmarks[11]
    right = landmarks[12]
    dis = (right.x - left.x)
    return dis

# 얼굴 거리 = 얼굴 세로 길이 기준
def get_face_distance(landmarks, image_shape):
    image_height = image_shape[0]
    nose_tip = landmarks[1]
    chin = landmarks[152]
    dy = abs(chin.y - nose_tip.y) * image_height
    return dy

def create_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
