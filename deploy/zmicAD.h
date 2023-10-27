#include <opencv2/core.hpp>

#ifndef _ZMIC_AD_H_
#define _ZMIC_AD_H_
// 声明read_raw
cv::Mat readRawImage(const std::string& path, bool twoView);
// 声明slice manager
class SliceManager 
{
private:
    int slice_size;
    double overlap_ratio;
    int fill;
public:
    SliceManager(int sliceSize, double overlapRatio, int fill);
    std::vector<cv::Mat> padded_slice(const cv::Mat& input, int& paddedH, int& paddedW, cv::Rect& origBox, int& numSlicesH, int& numSlicesW);
    cv::Mat merged_crop(const std::vector<cv::Mat>& slices, int paddedH, int paddedW, const cv::Rect& origBox, int numSlicesH, int numSlicesW);
};
// 声明channeled norm
cv::Mat channeled_norm(const cv::Mat& grayMat, std::string mode);
// 声明pre process
void meanStdNorm(cv::Mat& input, const std::vector<double>& mean, const std::vector<double>& std);
void matToFloatArray(const cv::Mat& mat, float* dst, int size);
// 声明post process
cv::Mat floatArrayToMat(const float* output, int numChannels, int numRows, int numCols);
struct VIXProb {
    int index;
    float prob;
    char label[64];
    float x1;
    float x2;
    float x3;
    float x4;
};
std::vector<VIXProb> postProcess(const cv::Mat& outputMat, double bin_thresh);
#endif