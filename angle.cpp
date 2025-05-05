#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

// 각도 계산 함수
float calculate_angle(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) {
    auto get_angle = [](const cv::Point2f& p1, const cv::Point2f& p2) {
        return std::atan2(p1.y - p2.y, p1.x - p2.x);
    };
    float angle1 = get_angle(a, b);
    float angle2 = get_angle(c, b);
    return std::abs((angle1 - angle2) * 180.0f / CV_PI);
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    // // 예시 랜드마크 좌표 (normalized)
    // cv::Point2f left_shoulder(0.3f, 0.4f);
    // cv::Point2f right_shoulder(0.7f, 0.4f);
    // cv::Point2f nose(0.5f, 0.2f);

        // 임시 좌표 추정 (픽셀 단위 기반, 좌우 반전 적용됨)
        cv::Point2f left_shoulder, right_shoulder, nose;

    cv::Mat frame;
    while (cap.read(frame)) {
        // 좌우 반전
        cv::flip(frame, frame, 1);

        // 가상의 인식 좌표 시뮬레이션 (예: 프레임 크기 기준 고정 위치)
        left_shoulder = cv::Point2f(0.3f * frame.cols, 0.4f * frame.rows);
        right_shoulder = cv::Point2f(0.7f * frame.cols, 0.4f * frame.rows);
        nose = cv::Point2f(0.5f * frame.cols, 0.2f * frame.rows);

        // 좌우 반전 좌표 보정 (x 좌표만 반전)
        left_shoulder.x = frame.cols - left_shoulder.x;
        right_shoulder.x = frame.cols - right_shoulder.x;
        nose.x = frame.cols - nose.x;
        

        // 중간 어깨 좌표 계산
        cv::Point2f mid_shoulder((left_shoulder.x + right_shoulder.x) / 2,
                                 (left_shoulder.y + right_shoulder.y) / 2);

        // 각도 계산
        float neck_angle = calculate_angle(left_shoulder, mid_shoulder, nose);
        // float shoulder_angle = calculate_angle(cv::Point2f(left_shoulder.x, left_shoulder.y - 0.1f),
        float shoulder_angle = calculate_angle(cv::Point2f(left_shoulder.x, left_shoulder.y - 0.1f * frame.rows),
                                               left_shoulder, right_shoulder);

        // 시각화
        // cv::circle(frame, cv::Point(left_shoulder.x * frame.cols, left_shoulder.y * frame.rows), 5, cv::Scalar(0, 255, 0), -1);
        // cv::circle(frame, cv::Point(right_shoulder.x * frame.cols, right_shoulder.y * frame.rows), 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(frame, left_shoulder, 5, cv::Scalar(0, 255, 0), -1);
        cv::circle(frame, right_shoulder, 5, cv::Scalar(0, 0, 255), -1);

        cv::putText(frame, "Neck Angle: " + std::to_string(int(neck_angle)), {30, 50},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Shoulder Angle: " + std::to_string(int(shoulder_angle)), {30, 80},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Posture Monitor", frame);
        if (cv::waitKey(5) == 27) break; // ESC 키 종료
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}