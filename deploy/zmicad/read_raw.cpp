#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>

void hexVis(uint16_t value) {
    uint8_t* bytePtr = reinterpret_cast<uint8_t*>(&value);
    std::cout << std::hex << 127 << std::endl;

    for (int i = 0; i < sizeof(uint16_t); i++) {
        std::cout << std::hex << static_cast<int>(bytePtr[i]) << " ";
    }

    std::cout << std::dec << std::endl;
}

cv::Mat readRawImage(const std::string& rawPath, bool twoView = false) {
    std::ifstream file(rawPath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open raw file.");
    }
    uint32_t nRow, nCol;
    uint8_t nBytes = 0;
    file.read(reinterpret_cast<char*>(&nRow), sizeof(nRow));
    file.read(reinterpret_cast<char*>(&nCol), sizeof(nCol));
    file.read(reinterpret_cast<char*>(&nBytes), sizeof(nBytes));
    // cout << nRow << "-" << nCol << "-" << static_cast<int>(nBytes) << endl;
    std::vector<uint16_t> imgData(nRow * nCol);
    if (nBytes == 2) {
        std::cout << "reading raw... raw image saved in 2 bytes per pixel" << std::endl;
        file.read(reinterpret_cast<char*>(imgData.data()), nRow * nCol * sizeof(uint16_t));
    } else {
        std::cout << "reading raw... raw image saved in 1 bytes per pixel" << std::endl;
        std::vector<uint8_t> imgBuffer(nRow * nCol);
        file.read(reinterpret_cast<char*>(imgBuffer.data()), nRow * nCol * sizeof(uint8_t));
        for (int i = 0; i < nRow * nCol; i++) {
            imgData[i] = imgBuffer[i];
        }
    }
    // hexVis(imgData[0]);
    cv::Mat img(nRow, nCol, CV_16U, imgData.data());
    if (twoView) {
        img = img(cv::Range(img.rows / 2, img.rows), cv::Range(0, img.cols));
    }
    return img.clone();
}

// int main() {
//     std::string rawPath = "../../data/xray/raw/bubbles/NG_20221009_104356.483.raw";
//     cv::Mat img = readRawImage(rawPath, false);
//     if (!img.empty()) {
//         cout << img << endl;
//         cout << img.rows << " " << img.cols << " " << img.dims << " " << img.channels() << endl;
//     } else {
//         std::cerr << "Failed to read the raw image." << std::endl;
//     }
//     return 0;
// }
