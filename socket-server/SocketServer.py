import cv2
import socket
import struct
import os
import numpy as np
import PoseEstimation as pe

face_mesh = pe.create_face_mesh()

def receive_image(conn):
    size_data = conn.recv(4)
    if not size_data:
        return None
    img_size = struct.unpack('!I', size_data)[0]
    img_data = b''
    while len(img_data) < img_size:
        packet = conn.recv(img_size - len(img_data))
        if not packet:
            return None
        img_data += packet
    img_np = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    cv2.imwrite('img.png', image)
    return img_data

def process_image(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = int(np.mean(gray))
    return brightness  # 평균 밝기 (0 ~ 255)

def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', 9000))
    server.listen(1)
    print("🟢 서버 대기 중...")

    while True:
        conn, addr = server.accept()

        img_bytes = receive_image(conn)
        if img_bytes:
            brightness = process_image(img_bytes)

            img_np = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                yaw, pitch, roll = pe.get_head_pose(landmarks, image.shape[:2])
                shoulder_roll = pe.get_shoulder_roll(landmarks, image.shape[:2])
                shoulder_dis = pe.get_shoulder_distance(landmarks, image.shape[:2])
                distance = pe.get_face_distance(landmarks, image.shape[:2])
                blink = pe.get_blink(landmarks, image.shape[:2])
                print(shoulder_dis)
                result = pe.estimate_final_pose(yaw, pitch, roll, distance, shoulder_roll, shoulder_dis, brightness, blink)
                # just test for blink
                # result = blink
            else:
                print("❌ 얼굴 인식 실패")
                result = 999
            conn.send(struct.pack('!I', result))
        else:
            print("❌ 이미지 수신 실패")

        conn.close()

if __name__ == "__main__":
    os.environ['GLOG_minloglevel'] = '2'
    run_server()