#include<torch/torch.h>
#include<torch/script.h>
#include<iostream>
#include <opencv2/opencv.hpp>

// 图像预处理
cv::Mat transformImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(224, 224));

    cv::Mat rotatedImage;
    cv::Point2f center(resizedImage.cols / 2.0, resizedImage.rows / 2.0);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, 10, 1.0);
    cv::warpAffine(resizedImage, rotatedImage, rotationMatrix, resizedImage.size());

    cv::Mat floatImage;
    rotatedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    cv::Mat normalizedImage;
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar stdDev(0.229, 0.224, 0.225);
    cv::subtract(floatImage, mean, normalizedImage);
    cv::divide(normalizedImage, stdDev, normalizedImage);

    cv::Mat transposedImage = normalizedImage.t();
    cv::Mat rgbImage;
    cv::cvtColor(transposedImage, rgbImage, cv::COLOR_BGR2RGB);

    return rgbImage;
}

int main() {
    std::string imagePath = "F:\\idcard\\card\\lhalfmask\\2023-08-11_14_24_02_front (10).bmp";

    cv::Mat rgbImage = transformImage(imagePath);

    torch::Tensor tensor_image = torch::from_blob(rgbImage.data, { 1,rgbImage.rows, rgbImage.cols,3 }, torch::kFloat);

    tensor_image = tensor_image.permute({ 0,3,1,2 });

    torch::jit::script::Module module;

    module = torch::jit::load("F:/model.pt");  //加载模型

    torch::DeviceType device_type = at::kCPU;    

    module.to(device_type);

    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(tensor_image.to(device_type));  //输入数据

    at::Tensor output = module.forward(inputs).toTensor();         
    std::cout << output << std::endl;                              

    auto maxResult = torch::max(output.data(), 1);
    auto maxValues = std::get<0>(maxResult);
    auto maxIndices = std::get<1>(maxResult);

    // 将索引从 Torch 的 LongTensor 转化为 C++ 的标准整数类型
    int predicted = maxIndices.item<int>();

    // 根据预测结果打印相应的标签
    if (predicted == 0) {
        std::cout << "正面" << std::endl;
    }
    else if (predicted == 1) {
        std::cout << "反面" << std::endl;
    }
    else {
        std::cout << "遮挡过度" << std::endl;
    }

    return 0;
}