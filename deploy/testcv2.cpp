#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/**
 * 常用函数：
 * Mat创建矩阵
 * cvtColor(src, dst, COLOR_XXX2XXX)
 * Mat::zeros()创建全0矩阵 **/

int main() {
    // Mat img = imread("/home/caoxiatain/DRAEM/test_output/gray_ms1k_80k/compose.png");
    // cout << img.dims << endl;
    // cout << img.cols << endl;
    // cout << img.rows << endl;
    // cout << img << endl;
    // int sz[2] = {100, 100};
    // Mat fgCount = Mat::zeros(2, sz, CV_8UC1);
    // Mat fgCount = Mat(10, 10, CV_8UC1, Scalar::all(0));
    // Mat retMat = Mat(6, 8, CV_8UC3, Scalar::all(255));
    // cout << fgCount.rows << " " << fgCount.cols << " " << fgCount.channels() << endl;
    // cout << retMat.rows << " " << retMat.cols << " " << retMat.channels() << endl;
    // cout << fgCount << endl;
    // cout << retMat << endl;
    Mat img = imread("../test_output/gray_ms1k_80k/compose.png");
    cout << img.rows << " " << img.cols << " " << img.dims << " " << img.channels() << endl;
    return 0;
}