#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

// Function to post-process the score map
std::vector<cv::Rect> post_process(const cv::Mat& score_map, double bin_thresh = 0.5, double pos_thresh = 0.3) {
    // Binary thresholding
    cv::Mat binary_mask = (score_map > bin_thresh) / 255;

    // Label connected components
    cv::Mat labeled_image, stats, centroids;
    int num_objects = cv::connectedComponentsWithStats(binary_mask, labeled_image, stats, centroids);

    // Calculate confidence scores for each bounding box
    std::vector<cv::Rect> bounding_boxes;
    for (int i = 1; i < num_objects; ++i) {
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        cv::Mat mask = (labeled_image == i);
        cv::Scalar mean_score = cv::mean(score_map, mask);

        // Calculate the confidence score for the bounding box
        double confidence = mean_score[0];

        // Check if the confidence score is above the positive threshold
        if (confidence > pos_thresh) {
            bounding_boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    return bounding_boxes;
}

int main() {
    // Load your score map here (score_map should be a grayscale image)
    cv::Mat score_map = cv::imread("score_map.jpg", cv::IMREAD_GRAYSCALE);

    // Define binary and positive thresholds
    double bin_thresh = 0.5;
    double pos_thresh = 0.3;

    // Perform post-processing
    std::vector<cv::Rect> positive_boxes = post_process(score_map, bin_thresh, pos_thresh);

    // Output the positive bounding boxes
    for (const cv::Rect& box : positive_boxes) {
        std::cout << "Positive Bounding Box: (" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << ")\n";
    }

    return 0;
}
