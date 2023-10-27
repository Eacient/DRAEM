// mean_std_norm Mat -> void
// mat2float array vector<Mat> -> vector<float>[bs * c * h * w]
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>

void meanStdNorm(cv::Mat& image, const std::vector<double>& channelMeans, const std::vector<double>& channelStdDevs) {
    assert(!image.empty() && image.channels() == channelMeans.size() && channelMeans.size() == channelStdDevs.size());
    int numChannels = image.channels();
    std::vector<cv::Mat> channels(numChannels);
    cv::split(image, channels);
    // 遍历每个通道
    for (int c = 0; c < numChannels; c++) {
        // 取出通道 c 的均值和标准差
        double mean = channelMeans[c];
        double stdDev = channelStdDevs[c];
        // std::cout << channels[c] << std::endl;
        // 将像素值标准化为均值为0、方差为1
        channels[c] = (channels[c] - mean) / stdDev;
    }
    return cv::merge(channels, image);
}


void matToFloatArray(const cv::Mat& mat, float* dst, int size) {
    assert(!mat.empty());

    int numChannels = mat.channels();
    int numRows = mat.rows;
    int numCols = mat.cols;

    assert(numChannels * numRows * numCols == size);

    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            const float* pixelPtr = mat.ptr<float>(row,col);
            for (int c = 0; c < numChannels; c++) {
                int arrayIndex = c * (numRows * numCols) + row * numCols + col;
                dst[arrayIndex] = pixelPtr[c]; // Assuming 32f format
            }
        }
    }
}



// int main() {
//     // 创建一个 10x10 的矩阵（图像），并用随机数据填充
//     cv::Mat image(10, 10, CV_32FC3);
//     randu(image, cv::Scalar(0), cv::Scalar(1));
//     std::cout << image << std::endl;
//     std::cout << image.rows << " " << image.cols << " " << image.channels() << std::endl;

//     std::vector<double> channelMeans = {1, 1, 0}; // 通道均值
//     std::vector<double> channelStdDevs = {1, 1, 1}; // 通道标准差

//     meanStdNorm(image, channelMeans, channelStdDevs);

//     std::cout << image << std::endl;
//     const float* pixelPtr = image.ptr<float>(3,3);
//     std::cout << pixelPtr[1] << std::endl;
//     return 0;
// }
