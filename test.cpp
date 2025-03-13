#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    cv::VideoCapture cap;

    std::cout << "Waiting for camera permission...\n";

    // 카메라 권한이 허용될 때까지 대기하는 루프
    while (!cap.isOpened()) {
        cap.open(0);
        if (cap.isOpened()) break;
        
        std::cerr << "Camera not accessible. Waiting for permission...\n";
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 1초 대기 후 다시 시도
    }

    std::cout << "Camera opened successfully!\n";

    // 프레임 크기 조절 (해상도 낮추기)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);  // 가로 320px
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240); // 비율 유지하여 세로 자동 조정
    cap.set(cv::CAP_PROP_FPS, 30); // 프레임 속도 제한

    // OpenCV 창 생성 (크기 조정 가능)
    cv::namedWindow("Webcam Stream", cv::WINDOW_NORMAL);
    cv::resizeWindow("Webcam Stream", 320, 240);  // 창 크기 강제 설정 (가로 320px)

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed!" << std::endl;
            break;
        }

        // 최신 프레임 출력
        cv::flip(frame, frame, 1);
        cv::imshow("Webcam Stream", frame);

        // 프레임 속도 제한 (30FPS 이하로 유지)
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // 30FPS (1000ms / 30)

        // 'q' 눌러서 종료
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // 프로그램 종료 시퀸스
    cap.release();
    cv::destroyAllWindows();
    return 0;
}