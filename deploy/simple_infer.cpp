#include <fstream> 
#include <iostream> 
#include <vector>
 
#include <NvInfer.h>
#include <TensorRT/samples/common/logger.h>
#include <zmicAD.h>
#include <opencv2/core.hpp>
 
#define CHECK(status)\
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0) 
 
using namespace nvinfer1; 
 

// const char* IN_NAME = "input"; 
// const char* OUT_NAME = "output"; 
const char* RAW_PATH = "test.raw";
const char* MODE = "ms";
static const std::vector<double> MEAN= {0.99310124, 0.9728329, 0.94237832, 0.90898088, 0.83903737, 0.74311205, 0.60096486, 0.50729982};
static const std::vector<double> STDEV= {0.05002454, 0.1179261,  0.18425917, 0.24282377, 0.29716158, 0.34985626, 0.38834061, 0.41779612};
static const double BIN_THRESH = 0.3;
static const int IN_H = 640; 
static const int IN_W = 640; 
static const int IN_C = 8;
static const int OUT_H = 640; 
static const int OUT_W = 640; 
static const int OUT_C = 2;
static const int BATCH_SIZE = 1; 
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 

void show_dims(nvinfer1::Dims dims) {
    for (int d = 0; d < dims.nbDims; d ++)
    {
        std::cout << dims.d[d] << " ";
    }
    std::cout << std::endl;
    return;
}

nvinfer1::Dims create_dims4(int b, int c, int h, int w) {
    nvinfer1::Dims dims;
    dims.nbDims = 4;
    dims.d[0] = b;
    dims.d[1] = c;
    dims.d[2] = h;
    dims.d[3] = w;
    return dims;
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) 
{
    const ICudaEngine& engine = context.getEngine(); 

    void* buffers[2]; 

    // 申请gpu内存
    CHECK(cudaMalloc(&buffers[0], batchSize * IN_C * IN_H * IN_W * sizeof(float))); 
    CHECK(cudaMalloc(&buffers[1], batchSize * OUT_C * OUT_H * OUT_W * sizeof(float))); 

    // 创建输入输出流
    cudaStream_t stream; 
    CHECK(cudaStreamCreate(&stream)); 

    // 输入、推理、输出
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * IN_C * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream)); 
    context.enqueue(batchSize, buffers, stream, nullptr); 
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUT_C * OUT_H * OUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream)); 
    cudaStreamSynchronize(stream); 

    // 释放流和缓存
    cudaStreamDestroy(stream); 
    CHECK(cudaFree(buffers[0])); 
    CHECK(cudaFree(buffers[1])); 
}

int main() 
{
	sample::Logger gLogger;
	// 1. 初始化TensorRT
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	assert(runtime != nullptr);

	// 2. 从引擎文件中加载TensorRT引擎
    nvinfer1::ICudaEngine* engine = nullptr;
    std::ifstream engineFile("bubbles.engine", std::ios::binary);
    if (engineFile.good()) {
        engineFile.seekg(0, engineFile.end);
        size_t fileSize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineData(fileSize);
        engineFile.read(engineData.data(), fileSize);
        engine = runtime->deserializeCudaEngine(engineData.data(), fileSize, nullptr);
    } else {
        std::cerr << "Failed to open engine file." << std::endl;
        return 1;
    }
	assert(engine != nullptr);
    std::cout << "input dimensions:" << std::endl;
    show_dims(engine->getBindingDimensions(0));
    std::cout << "output dimensions:" << std::endl;
	show_dims(engine->getBindingDimensions(1));

	// 3. 创建执行上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);
    nvinfer1::Dims inputdims = create_dims4(BATCH_SIZE, IN_C, IN_H, IN_W);
    context->setBindingDimensions(0, inputdims);
    // nvinfer1::Dims outputdims = create_dims4(BATCH_SIZE, OUT_C, OUT_H, OUT_W);
    // context->setBindingDimensions(1, outputdims);
 
	// 4. 数据处理和推理
    SliceManager sliceManager = SliceManager(IN_H, 0.25, 65000);
	// float* inputData = new float[BATCH_SIZE * IN_C * IN_H * IN_W];
    // 4.1. 读取raw -> Mat 单通道65000
    cv::Mat raw = readRawImage(RAW_PATH, false);
    // 4.2. slice -> Mat -> vector <Mat> 65000
    int padded_h, padded_w;
    cv::Rect orig_box;
    int n_slice_h, n_slice_w;
    std::vector<cv::Mat> rawSlices = sliceManager.padded_slice(raw, padded_h, padded_w, orig_box, n_slice_h, n_slice_w);
    int numSlices = rawSlices.size();
    // 4.3. channeled_norm -> 逐Mat处理，单通道->八通道，取值0-1
    std::vector<cv::Mat> inputSlices;
    for (auto s : rawSlices) {
        inputSlices.push_back(channeled_norm(s, MODE));
    }
    // std::cout << input[0].channels() << std::endl;
    // 4.4. pre_process -> 逐Mat处理，通道归一化，序列化为float数组
    float* inputArrays = new float[numSlices * IN_C * IN_H * IN_W];
    int inputInd = 0;
    int inputSize = IN_C * IN_H * IN_W;
    for (auto s : inputSlices) {
        meanStdNorm(s, MEAN, STDEV);
        matToFloatArray(s, inputArrays, inputSize);
        inputInd += inputSize;
    }
    // 4.5. 创建对应的输出缓存 vector <Mat>
    float* outputData = new float[BATCH_SIZE * OUT_C * OUT_H * OUT_W]; 
    std::vector<cv::Mat> outputMats;
    int outputSize = OUT_C * OUT_H * OUT_W;
    // 4.6. 循环推理
    for (int i = 0; i < numSlices; i++) {
        // gpu推理
        doInference(*context, inputArrays+i*inputSize, outputData, BATCH_SIZE); 
        // 填充输出数组
        outputMats.push_back(floatArrayToMat(outputData, OUT_C, OUT_H, OUT_W));
    }
    // 4.7. merge -> Mat
    cv::Mat mergedProb = sliceManager.merged_crop(outputMats, padded_h, padded_w, orig_box, n_slice_h, n_slice_w);
    // 4.8. post process -> VixProb
    std::vector<VIXProb> vixOutput = postProcess(mergedProb, BIN_THRESH);

    for (const auto& box : vixOutput) {
        std::cout << "Positive Bounding Box: (" << box.x1 << ", " << box.x2 << ", " << box.x3 << ", " << box.x4 << " " << box.prob << ")\n";
    }

    // 5. 释放资源
	// delete[] inputData;
    delete[] inputArrays;
	delete[] outputData;
    context->destroy(); 
    engine->destroy(); 
    runtime->destroy(); 

    return 0; 
} 
