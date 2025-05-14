import cv2
import socket
import struct
import os
import numpy as np
import PoseEstimation as pe
import time

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
    return img_data

def process_image(img_bytes):
    return len(img_bytes) % 1000  # 예시

def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 9000))
    server.listen(1)
    print("🟢 서버 대기 중...")

    while True:
        conn, addr = server.accept()

        img_bytes = receive_image(conn)
        if img_bytes:
            img_np = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                yaw, pitch, roll = pe.get_head_pose(landmarks, image.shape[:2])
                shoulder_roll = pe.get_shoulder_roll(landmarks, image.shape[:2])
                distance = pe.get_face_distance(landmarks, image.shape[:2])
                blink = pe.get_blink(landmarks, image.shape[:2])
                print(f"\r🎯 Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}, Shoulder: {shoulder_roll:.2f}, Distance: {distance:.2f}")
                # result = int(abs(yaw + pitch + roll + shoulder_roll + distance)) % 1000
                result = blink
            else:
                print("❌ 얼굴 인식 실패")
                result = 999
            conn.send(struct.pack('!I', result))
        else:
            print("❌ 이미지 수신 실패")

        conn.close()

def send_webcam_images():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임 읽기 실패")
                break

            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()

            try:
                client.connect(('localhost', 9000))
                client.send(struct.pack('!I', len(img_bytes)))
                client.sendall(img_bytes)

                result_data = client.recv(4)
                result = struct.unpack('!I', result_data)[0]
                print("📨 서버 응답 결과:", result)

                client.close()
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            except Exception as e:
                print("❌ 전송 실패:", e)

            time.sleep(0.05)

    finally:
        cap.release()

if __name__ == "__main__":
    os.environ['GLOG_minloglevel'] = '2'
    # run_server()  # 주석 처리
    send_webcam_images()