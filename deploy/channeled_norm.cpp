#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat channeled_norm(const cv::Mat& image, const std::string& mode) {
    // image [h, w], 0-65000
    assert(mode == "single" || mode == "ms" || mode == "sw");

    cv::Mat normalized_image;

    if (mode == "single") {
        image.convertTo(normalized_image, CV_32F, 1.0 / 65535.0);
    } else if (mode == "ms") {
        std::vector<int> ths = {511, 1023, 2047, 4095, 8191, 16383, 32767, 65535};
        std::vector<cv::Mat> channels;

        for (int th : ths) {
            cv::Mat channel = cv::Mat::zeros(image.size(), CV_32F);
            cv::Mat mask = (image >= th);
            image.copyTo(channel, mask);
            cv::multiply(channel, 1.0 / th, channel);
            channels.push_back(channel);
        }

        cv::vconcat(channels, normalized_image);
    } else {
        std::vector<int> lows = {0, 256, 512, 1024, 2048, 4096, 8192, 16384};
        std::vector<int> highs = {511, 1023, 2047, 4095, 8191, 16383, 32767, 65535};
        std::vector<cv::Mat> channels;

        for (int i = 0; i < lows.size(); i++) {
            cv::Mat channel = cv::Mat::zeros(image.size(), CV_32F);
            cv::Mat mask = (image >= highs[i]);
            image.copyTo(channel, mask);
            mask = (image < highs[i]) & (image > lows[i]);
            cv::multiply(image - lows[i], 1.0 / (highs[i] - lows[i]), channel, mask);
            channels.push_back(channel);
        }

        cv::vconcat(channels, normalized_image);
    }

    return normalized_image;
}

int main() {
    cv::Mat input_image = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat normalized_image = channeled_norm(input_image, "single"); // Change mode as needed
    return 0;
}