#ifndef _SLICE_MANAGER_H_
#define _SLICE_MANAGER_H_
#include <opencv2/core.hpp>
class SliceManager 
{
private:
    int slice_size;
    double overlap_ratio;
    int fill;
public:
    SliceManager(int, double, int);
    std::vector<cv::Mat> padded_slice(const cv::Mat&, int&, int&, cv::Rect&, int&, int&);
    cv::Mat merged_crop(const std::vector<cv::Mat>&, int, int, const cv::Rect&, int, int);
};
#endif