#include <iostream>
#include <vector>
#include <slice_manager.h>
#include <opencv2/core.hpp>

SliceManager::SliceManager(int slice_size = 640, double overlap_ratio = 0.25, int fill = 65000)
    : slice_size(slice_size), overlap_ratio(overlap_ratio), fill(fill) {}

std::vector<cv::Mat> SliceManager::padded_slice(const cv::Mat& img, int& padded_h, int& padded_w, cv::Rect& orig_box, int& n_slice_h, int& n_slice_w) 
{
    int orig_h = img.rows;
    int orig_w = img.cols;
    int numChannels = img.channels();
    std::vector<cv::Mat> slices;

    orig_box = cv::Rect(0, 0, orig_w, orig_h);
    int padding_h = (orig_h > slice_size) ? 0 : slice_size - orig_h;
    int padding_w = (orig_w > slice_size) ? 0 : slice_size - orig_w;

    cv::Mat paddedImage;
    if (padding_h > 0 || padding_w > 0) {
        int st_h = padding_h / 2;
        int st_w = padding_w / 2;
        orig_box = cv::Rect(st_w, st_h, orig_w, orig_h);
        cv::Mat bg = cv::Mat::ones(orig_h + padding_h, orig_w + padding_w, CV_32FC(numChannels)) * fill;
        img.copyTo(bg(orig_box));
        paddedImage = bg;
    } else {
        paddedImage = img;
    }

    padded_h = orig_h + padding_h;
    padded_w = orig_w + padding_w;

    std::vector<int> slice_h_sts, slice_w_sts;
    for (int i = 0; i < padded_h - slice_size; i += slice_size * (1 - overlap_ratio)) {
        slice_h_sts.push_back(i);
    }
    slice_h_sts.push_back(padded_h - slice_size);
    for (int i = 0; i < padded_w - slice_size; i += slice_size * (1 - overlap_ratio)) {
        slice_w_sts.push_back(i);
    }
    slice_w_sts.push_back(padded_w - slice_size);

    n_slice_h = slice_h_sts.size();
    n_slice_w = slice_w_sts.size();

    for (int h_st : slice_h_sts) {
        for (int w_st : slice_w_sts) {
            cv::Mat slice = paddedImage(cv::Rect(w_st, h_st, slice_size, slice_size));
            slices.push_back(slice);
        }
    }

    return slices;
}

cv::Mat SliceManager::merged_crop(const std::vector<cv::Mat>& pred_slices, int padded_h, int padded_w, const cv::Rect& orig_box, int n_slice_h, int n_slice_w) {
    int numChannels = pred_slices[0].channels();
    cv::Mat bg = cv::Mat::zeros(padded_h, padded_w, CV_32FC(numChannels));
    cv::Mat cntMap = cv::Mat::zeros(padded_h, padded_w, CV_8UC1);

    int h_st = 0, w_st = 0, ind = 0;
    int h_gap = slice_size * (1 - overlap_ratio);
    int w_gap = slice_size * (1 - overlap_ratio);

    for (int i = 0; i < n_slice_h; ++i) {
        w_st = 0;
        for (int j = 0; j < n_slice_w; ++j) {
            int h_end = h_st + slice_size;
            if (h_end > padded_h) {
                h_st = padded_h - slice_size;
                h_end = padded_h;
            }
            int w_end = w_st + slice_size;
            if (w_end > padded_w) {
                w_st = padded_w - slice_size;
                w_end = padded_w;
            }
            cv::Rect slice_rect(w_st, h_st, slice_size, slice_size);
            (bg(slice_rect)) += pred_slices[ind];
            cntMap(slice_rect) += 1;
            w_st += w_gap;
            ind += 1;
        }
        h_st += h_gap;
    }

    // std::cout << cntMap << std::endl;
    std::vector<cv::Mat> bgChannels;
    cv::split(bg, bgChannels);
    // std::cout << bg << std::endl;
    for (auto bgChannel : bgChannels) {
        bgChannel /= cntMap;
    }
    cv::merge(bgChannels, bg);
    // std::cout << bg << std::endl;
    return bg(orig_box);
}


// int main() {
//     SliceManager slice_manager(2, 0.25, 1);
//     cv::Mat input_img = cv::Mat(5, 5, CV_32FC4, cv::Scalar::all(1));

//     int padded_h, padded_w;
//     cv::Rect orig_box;
//     int n_slice_h, n_slice_w;

//     std::vector<cv::Mat> slices = slice_manager.padded_slice(input_img, padded_h, padded_w, orig_box, n_slice_h, n_slice_w);
//     // Process the slices
//     cv::Mat output_img = slice_manager.merged_crop(slices, padded_h, padded_w, orig_box, n_slice_h, n_slice_w);
//     return 0;
// }


// class SliceManager {
// public:
//     SliceManager(int slice_size = 640, double overlap_ratio = 0.25, int fill = 65000)
//         : slice_size(slice_size), overlap_ratio(overlap_ratio), fill(fill) {}

//     std::vector<cv::Mat> padded_slice(const cv::Mat& img, int& padded_h, int& padded_w, cv::Rect& orig_box, int& n_slice_h, int& n_slice_w) {
//         int orig_h = img.rows;
//         int orig_w = img.cols;
//         int numChannels = img.channels();
//         std::vector<cv::Mat> slices;

//         orig_box = cv::Rect(0, 0, orig_w, orig_h);
//         int padding_h = (orig_h > slice_size) ? 0 : slice_size - orig_h;
//         int padding_w = (orig_w > slice_size) ? 0 : slice_size - orig_w;

//         cv::Mat paddedImage;
//         if (padding_h > 0 || padding_w > 0) {
//             int st_h = padding_h / 2;
//             int st_w = padding_w / 2;
//             orig_box = cv::Rect(st_w, st_h, orig_w, orig_h);
//             cv::Mat bg = cv::Mat::ones(orig_h + padding_h, orig_w + padding_w, CV_32FC(numChannels)) * fill;
//             img.copyTo(bg(orig_box));
//             paddedImage = bg;
//         } else {
//             paddedImage = img;
//         }

//         padded_h = orig_h + padding_h;
//         padded_w = orig_w + padding_w;

//         std::vector<int> slice_h_sts, slice_w_sts;
//         for (int i = 0; i < padded_h - slice_size; i += slice_size * (1 - overlap_ratio)) {
//             slice_h_sts.push_back(i);
//         }
//         slice_h_sts.push_back(padded_h - slice_size);
//         for (int i = 0; i < padded_w - slice_size; i += slice_size * (1 - overlap_ratio)) {
//             slice_w_sts.push_back(i);
//         }
//         slice_w_sts.push_back(padded_w - slice_size);

//         n_slice_h = slice_h_sts.size();
//         n_slice_w = slice_w_sts.size();

//         for (int h_st : slice_h_sts) {
//             for (int w_st : slice_w_sts) {
//                 cv::Mat slice = paddedImage(cv::Rect(w_st, h_st, slice_size, slice_size));
//                 slices.push_back(slice);
//             }
//         }

//         return slices;
//     }

//     cv::Mat merged_crop(const std::vector<cv::Mat>& pred_slices, int padded_h, int padded_w, const cv::Rect& orig_box, int n_slice_h, int n_slice_w) {
//         int numChannels = pred_slices[0].channels();
//         cv::Mat bg = cv::Mat::zeros(padded_h, padded_w, CV_32FC(numChannels));
//         cv::Mat cntMap = cv::Mat::zeros(padded_h, padded_w, CV_8UC1);

//         int h_st = 0, w_st = 0, ind = 0;
//         int h_gap = slice_size * (1 - overlap_ratio);
//         int w_gap = slice_size * (1 - overlap_ratio);

//         for (int i = 0; i < n_slice_h; ++i) {
//             w_st = 0;
//             for (int j = 0; j < n_slice_w; ++j) {
//                 int h_end = h_st + slice_size;
//                 if (h_end > padded_h) {
//                     h_st = padded_h - slice_size;
//                     h_end = padded_h;
//                 }
//                 int w_end = w_st + slice_size;
//                 if (w_end > padded_w) {
//                     w_st = padded_w - slice_size;
//                     w_end = padded_w;
//                 }
//                 cv::Rect slice_rect(w_st, h_st, slice_size, slice_size);
//                 (bg(slice_rect)) += pred_slices[ind];
//                 cntMap(slice_rect) += 1;
//                 w_st += w_gap;
//                 ind += 1;
//             }
//             h_st += h_gap;
//         }

//         // std::cout << cntMap << std::endl;


//         std::vector<cv::Mat> bgChannels;
//         cv::split(bg, bgChannels);
//         // std::cout << bg << std::endl;
//         for (auto bgChannel : bgChannels) {
//             bgChannel /= cntMap;
//         }
//         cv::merge(bgChannels, bg);
//         // std::cout << bg << std::endl;
//         return bg(orig_box);
//     }

// private:
//     int slice_size;
//     double overlap_ratio;
//     int fill;
// };