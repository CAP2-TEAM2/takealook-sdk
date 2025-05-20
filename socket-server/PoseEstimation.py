import numpy as np
import mediapipe as mp
import time
from collections import deque

PITCH_OFFSET = -20
MIN_BRIGHTNESS = 150
CLOSE_DISTANCE = 30
SHOULDER_DEGREE = 10
LOOK_DOWN = 28
MIN_BLINK = 4
TURTLE_RANGE = 10

blink_times = deque()

TURTLE_INDEX = 15
turtle_q = deque(maxlen=TURTLE_INDEX)

# for init pose average
INIT_INDEX = 50  # 초기 기준값 인덱스 수
init_pose_samples = []
init_done = False

def estimate_final_pose(yaw, pitch, roll, distance, shoulder_roll, shoulder_dis, brightness, blink):
    global init_start_time, init_pose_samples, init_done
    # print(shoulder_dis)

    if len(init_pose_samples) <= INIT_INDEX:
        init_pose_samples.append((yaw, pitch, roll, distance, shoulder_roll, shoulder_dis, brightness, blink))
        return 0  # 초기에는 측정값 보내지 않음
    elif not init_done:
        # 평균 기준값 계산
        arr = np.array(init_pose_samples)
        avg_yaw, avg_pitch, avg_roll, avg_dist, avg_shoulder_roll, avg_shoulder_dis, avg_brightness, avg_blink = np.mean(arr, axis=0)
        estimate_final_pose.reference = {
            'brightness': avg_brightness,
            'distance': avg_dist,
            'blink': avg_blink,
            'shoulder_roll': avg_shoulder_roll,
            'roll': avg_roll,
            'shoulder_dis': avg_shoulder_dis,
            'turtle': avg_dist / avg_shoulder_dis * 1000
        }
        init_done = True

    ref = estimate_final_pose.reference

    # 밝기 / 얼굴 거리 / 눈 깜빡임 / 어깨, 턱 각도 / 거북목
    result = [0, 0, 0, 0, 0]
    result[0] = int(ref['brightness'] / brightness > 1.1) + 1
    result[1] = int(distance / ref['distance'] > 1.1) + 1
    result[2] = int(blink <= ref['blink']) + 1
    result[3] = int(abs(shoulder_roll - 90) > SHOULDER_DEGREE / 2 and abs(roll) > 20) * (int(shoulder_roll > 90) + 1) + 1
    # 거북목 판펼
    # result[4] = int(turtle <= ref['turtle']) + 1
    # result[4] = int(ref['turtle'] - distance / shoulder_dis * 1000 > TURTLE_RANGE) + 1
    t = distance/ shoulder_dis * 1000
    turtle_now = int(ref['turtle'] - t > TURTLE_RANGE) + 1
    turtle_q.append(turtle_now)

    if len(turtle_q) == TURTLE_INDEX and all(v == 2 for v in turtle_q):
        result[4] = 2
    else:
        result[4] = 1
    # for turtle neck debug
    # print(f"avg: {ref['turtle']}, now: {t}")
    # print(distance/(ref['shoulder_dis'] * 10))
    return int(''.join(map(str, result)))

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
    image_width = image_shape[1]
    left = landmarks[11]
    right = landmarks[12]
    dis = (left.x - right.x) * image_width
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

def create_pose():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

def get_blink(landmarks, image_shape):
    global blink_times
    image_height, image_width = image_shape

    # 눈 위/아래 랜드마크
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]

    # 눈 높이 계산
    left_eye_height = (left_eye_bottom.y - left_eye_top.y) * image_height
    right_eye_height = (right_eye_bottom.y - right_eye_top.y) * image_height

    threshold = 1.0  # 눈 감김 판정 기준
    now = time.time()

    # 눈이 감겼다고 판단되면 기록
    if left_eye_height < threshold and right_eye_height < threshold:
        # 직전 깜빡임과 0.1초 이상 차이날 때만 추가
        if not blink_times or now - blink_times[-1] > 0.1:
            blink_times.append(now)

    # 큐에서 10초 넘은 오래된 기록 제거
    while blink_times and now - blink_times[0] > 10.0:
        blink_times.popleft()

    # 현재 10초 이내 깜빡임 횟수 반환
    return len(blink_times)

# def get_turtle(shoulder_dis, face_dis):
    # return int(shoulder_dis / face_dis * 100000000)