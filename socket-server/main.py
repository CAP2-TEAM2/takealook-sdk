import cv2
import time
from pose_estimator import (
    get_head_pose,
    get_face_distance,
    get_shoulder_roll,
    create_face_mesh
)

def estimate_pose_from_frame(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_shape = (image.shape[0], image.shape[1])

    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return -1, None, None, None, None, None

    landmarks = results.multi_face_landmarks[0].landmark

    yaw, pitch, roll = get_head_pose(landmarks, image_shape)
    distance = get_face_distance(landmarks, image_shape)
    shoulder_roll = get_shoulder_roll(landmarks, image_shape)
    roll_deviation = abs(shoulder_roll - 90)

    if distance > 70:
        code = 20
    elif pitch < 26:
        code = 10
    elif roll_deviation > 5:
        code = 30
    else:
        code = 0

    return code, pitch, distance, shoulder_roll, yaw, roll

def classify_pose(code):
    if code == 0:
        return "정상 자세입니다."
    elif code == 10:
        return "고개를 숙이고 있습니다."
    elif code == 20:
        return "화면과 너무 가깝습니다."
    elif code == 30:
        return "어깨가 비대칭입니다."
    else:
        return "자세 상태를 확인할 수 없습니다."

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 열기 실패")
        exit()

    print("실시간 자세 분석 시작 (ESC 키로 종료)")
    last_print_time = 0

    with create_face_mesh() as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패")
                break

            frame = cv2.flip(frame, 1)

            code, pitch, distance, shoulder_roll, yaw, roll = estimate_pose_from_frame(frame, face_mesh)

            current_time = time.time()
            if current_time - last_print_time >= 3:
                if pitch is not None:
                    print(f"고개 각도: {pitch:.1f}°, 얼굴 거리(px): {distance:.1f}, 어깨 기울기: {shoulder_roll:.1f}°, yaw: {yaw:.1f}°, roll: {roll:.1f}°")
                    print(classify_pose(code))

                    # 좌우로 고개 돌림
                    if yaw < 90:
                        print("오른쪽으로 고개를 돌렸습니다.")
                    elif yaw > 98:
                        print("왼쪽으로 고개를 돌렸습니다.")

                    # 좌우로 기울이기
                    if roll <= -15:
                        print("왼쪽으로 고개를 기울였습니다.")
                    elif roll >= 16:
                        print("오른쪽으로 고개를 기울였습니다.")
                else:
                    print("얼굴을 인식할 수 없습니다.")
                last_print_time = current_time

            cv2.imshow("Pose Monitor", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
