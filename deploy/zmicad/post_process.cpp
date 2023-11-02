#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

struct VIXProb {
    int index;
    float prob;
    char label[64];
    float x1;
    float x2;
    float x3;
    float x4;
};

cv::Mat floatArrayToMat(const float* output, int numChannels, int numRows, int numCols) {
    // 创建一个空的Mat对象
    cv::Mat mat(numRows, numCols, CV_32FC(numChannels));
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            float* pixelPtr = mat.ptr<float>(row, col);
            for (int c = 0; c < numChannels; c++) {
                int arrayIndex = c * (numRows * numCols) + row * numCols + col;
                pixelPtr[c] = output[arrayIndex];
            }
        }
    }
    return mat;
}

// vector<float> [n * 2 * h *w]
// Function to post-process the score map
std::vector<VIXProb> postProcess(const cv::Mat& output_map, double bin_thresh = 0.5) {
    // 双通道结果转单通道
    std::vector<cv::Mat> outputChannels;
    cv::split(output_map, outputChannels);
    cv::Mat score_map = outputChannels[1];
    // Binary thresholding
    cv::Mat binary_mask = (score_map > bin_thresh) / 255;

    // Label connected components
    cv::Mat labeled_image, stats, centroids;
    int num_objects = cv::connectedComponentsWithStats(binary_mask, labeled_image, stats, centroids);

    // Calculate confidence scores for each bounding box
    std::vector<cv::Rect> bounding_boxes;
    std::vector<VIXProb> vixProbs(num_objects);
    for (int i = 1; i < num_objects; ++i) {
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        cv::Mat mask = (labeled_image == i);
        cv::Scalar mean_score = cv::mean(score_map, mask);

        // Calculate the confidence score for the bounding box
        double confidence = mean_score[0];
        vixProbs[i].prob = confidence;
        vixProbs[i].x1 = left;
        vixProbs[i].x2 = top;
        vixProbs[i].x3 = width;
        vixProbs[i].x4 = height;
    }
    return vixProbs;
}

// int main() {
//     // Load your score map here (score_map should be a grayscale image)
//     cv::Mat score_map = cv::Mat(10, 10, CV_32FC2);

//     // Define binary and positive thresholds
//     double bin_thresh = 0.5;

//     // Perform post-processing
//     std::vector<VIXProb> positive_boxes = postProcess(score_map, bin_thresh);

//     // Output the positive bounding boxes
//     for (const auto& box : positive_boxes) {
//         std::cout << "Positive Bounding Box: (" << box.x1 << ", " << box.x2 << ", " << box.x3 << ", " << box.x4 << " " << box.prob << ")\n";
//     }

//     return 0;
// }
