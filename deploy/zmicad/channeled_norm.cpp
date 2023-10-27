// 输入二维矩阵图像
// 返回0，1范围内的多通道图像
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>

using namespace std;
cv::Mat channeled_norm(const cv::Mat& image, std::string mode) {
    assert(mode == "single" || mode == "ms" || mode == "sw");
    cv::Mat image32F;
    cv::Mat retImage;
    image.convertTo(image32F, CV_32F);
    if (mode == "single") {
        retImage = image32F / 65535;
    } else if (mode == "ms") {
        vector<int> ths = {511, 1023, 2047, 4095, 8191, 16383, 32767, 65535};
        vector<cv::Mat> channels;
        for (int th : ths) {
            cv::Mat channel = cv::min(image32F, th) / th;
            // cv::Mat channel = cv::min(image, th);
            channels.push_back(channel);
        }
        cv::merge(channels, retImage);
    } else {
        vector<int> lows = {0, 256, 512, 1024, 2048, 4096, 8192, 16384};
        vector<int> highs = {511, 1023, 2047, 4095, 8191, 16383, 32767, 65535};
        vector<cv::Mat> channels;
        for (int i = 0; i < lows.size(); i++) {
            cv::Mat channel = (cv::min(cv::max(image32F, lows[i]), highs[i]) - lows[i]) / (highs[i] - lows[i]);
            // cv::Mat channel = (cv::min(cv::max(image, lows[i]), highs[i]) - lows[i]);
            channels.push_back(channel);
        }
        cv::merge(channels, retImage);
    }
    return retImage;
}

// int main() {
//     // 创建一个 10x10 的矩阵（图像），并用随机数据填充
//     cv::Mat image(10, 10, CV_16U);
//     randu(image, cv::Scalar(0), cv::Scalar(65000));
//     // 打印原始图像
//     cout << "Original Image:" << endl;
//     cout << image << endl;

//     // 使用 channeled_norm 函数对图像进行归一化（以 'single' 模式为例）
//     cv::Mat normalized_image_single = channeled_norm(image, "single");
//     // 打印归一化后的图像
//     cout << "Normalized Image (single):" << endl;
//     cout << normalized_image_single << endl;

//     // 使用 channeled_norm 函数对图像进行归一化（以 'ms' 模式为例）
//     cv::Mat normalized_image_ms = channeled_norm(image, "ms");
//     // 打印归一化后的图像
//     cout << "Normalized Image (ms):" << endl;
//     cout << normalized_image_ms.channels() << endl;
//     cout << normalized_image_ms << endl;

//     // 使用 channeled_norm 函数对图像进行归一化（以 'sw' 模式为例）
//     cv::Mat normalized_image_sw = channeled_norm(image, "sw");
//     // 打印归一化后的图像
//     cout << "Normalized Image (sw):" << endl;
//     cout << normalized_image_sw << endl;

//     return 0;
// }