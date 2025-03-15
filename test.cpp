#include <opencv2/opencv.hpp>
#include <iostream>

// 🔹 얼굴 검출을 위한 Haar Cascade 파일 경로
const std::string CASCADE_PATH = "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

int main() {
    cv::VideoCapture cap(0);

    while (!cap.isOpened()) {
        cap.open(0);
        if (cap.isOpened()) break;
        std::cerr << "Camera not accessible. Waiting for permission...\n";
    }

    std::cout << "Camera opened successfully!\n";

    // ✅ 해상도 최적화 (연산량 감소)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(cv::CAP_PROP_FPS, 30);

    // 🔹 OpenCV Cascade Classifier 로드
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(CASCADE_PATH)) {
        std::cerr << "Error: Could not load face cascade model!" << std::endl;
        return -1;
    }

    cv::namedWindow("Webcam Stream", cv::WINDOW_NORMAL);
    cv::resizeWindow("Webcam Stream", 320, 240);

    while (true) {
        cv::Mat frame, gray;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed!" << std::endl;
            continue;
        }

        // ✅ 좌우 반전 및 Grayscale 변환
        cv::flip(frame, frame, 1);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if (gray.empty()) continue;

        // 🔹 얼굴 검출
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

        // 🔹 얼굴을 네모 박스로 표시
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Webcam Stream", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}