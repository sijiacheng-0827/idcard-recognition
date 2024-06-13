#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <array>
#include <tuple>
#include <cstdint>
#include "idcard_interface.h"
#include "idcard.h"

int IdCard_Interface::idcard_dirty_detect(const char* front, const char* back, int* results) {
    class Range {
    public:
        int min;
        int max;
    };

    class Position {
    public:
        int top;
        int left;
        int right;
        int bottom;
    };

    // C++11 起支持 enum class
    enum class IdCardDirtyArea {
        AREA_0 = 0,
        AREA_1 = 1,
        AREA_2 = 2,
        AREA_3 = 3,
        AREA_4 = 4,
        AREA_5 = 5,
        AREA_6 = 6,
        AREA_7 = 7,
        AREA_8 = 8,
        AREA_9 = 9,
        AREA_10 = 10,
        AREA_11 = 11
    };

    enum class DirtyState {
        NOT_DIRTY = 0,
        DIRTY = 1
    };

    Range ranges_front_width[7] = { {0, 340}, {0, 173}, {174, 340}, {0, 380}, {0, 420}, {0, 640}, {410, 640} };
    Range ranges_front_hight[7] = { {1, 135}, {136, 220}, {136, 220}, {221, 320}, {321, 475}, {476, 640}, {20, 500} };
    Range ranges_back_width[3] = { {0, 155}, {0, 640}, {0, 640} };
    Range ranges_back_hight[3] = { {0, 355}, {400, 497}, {498, 640} };
    Position position = { -1, -1 };  // 初始化位置信息

    IdCard id_card;
    id_card.model_path = "/home/fortune/program_csj/idcard_interface/best.onnx";

    // Load model
    cv::dnn::Net net_front = cv::dnn::readNetFromONNX(id_card.model_path);             //0612
    cv::dnn::Net net_back = cv::dnn::readNetFromONNX(id_card.model_path);

    // Read card image
    cv::Mat front_img = cv::imread(front);  // read front card		
    cv::Mat back_img = cv::imread(back);  // read back card
    

    // Check if the image was successfully loaded
    if (front_img.empty() || back_img.empty()) {
        std::cerr << "Could not read one of the images." << std::endl;
        return -1;
    }

    // Adjust image size to match model input
    cv::resize(front_img, front_img, cv::Size(640, 640));
    cv::resize(back_img, back_img, cv::Size(640, 640));

    // Create network input
    cv::Mat front_blob = cv::dnn::blobFromImage(front_img, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    
    
    int rows = front_blob.rows;         // 高度
    int cols = front_blob.cols;         // 宽度
    int channels = front_blob.channels(); // 通道数
    std::cout << "Blob dimensions: "
        << "[batch size = " << front_blob.total() / (rows * cols * channels)
        << ", height = " << rows
        << ", width = " << cols
        << ", channels = " << channels << "]" << std::endl;

    cv::Mat back_blob = cv::dnn::blobFromImage(back_img, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);

    // Set network input
    net_front.setInput(front_blob);
    cv::Mat front_detection = net_front.forward();
    net_back.setInput(back_blob);
    cv::Mat back_detection = net_back.forward();

    // Filter the detection results on both the front and back sides
    cv::Mat front_filtered_boxes = id_card.filter_box(front_detection, 0.25, 0.45);
    cv::Mat back_filtered_boxes = id_card.filter_box(back_detection, 0.25, 0.45);

    
    // Obtain the bounding box coordinates of the front and back sides
    
    std::vector<std::tuple<int, int, int, int>> front_result = id_card.draw(front_img, front_filtered_boxes);
    
    // 遍历边界框并处理每个边界框
    //IdCardDirtyArea front_area;
    for (const auto& box : front_result) {
    int left, top, right, bottom;
    std::tie(left, top, right, bottom) = box;
    position.top = top;
    position.left = left;
    position.bottom = bottom;
    position.right = right;

    bool front_hightCondition[7];
    bool front_widthCondition[7];
    bool front_hightCondition1[7];
    bool front_widthCondition1[7];

    for (int i = 0; i < 7; ++i) {
        front_hightCondition[i] = (ranges_front_hight[i].max > position.top && ranges_front_hight[i].max < position.bottom) || 
                                  (ranges_front_hight[i].min > position.top && ranges_front_hight[i].min < position.bottom);

        front_widthCondition[i] = (ranges_front_width[i].max > position.left && ranges_front_width[i].max < position.right) || 
                                  (ranges_front_width[i].min > position.left && ranges_front_width[i].min < position.right);

        front_hightCondition1[i] = (position.bottom > ranges_front_hight[i].min && position.bottom < ranges_front_hight[i].max) || 
                                   (position.top > ranges_front_hight[i].min && position.top < ranges_front_hight[i].max);

        front_widthCondition1[i] = (position.right > ranges_front_width[i].min && position.right < ranges_front_width[i].max) || 
                                   (position.left > ranges_front_width[i].min && position.left < ranges_front_width[i].max);
    }

    // 检查条件并设置相应的 front_area
    for (int i = 0; i < 7; ++i) {
        if ((front_hightCondition[i] && front_widthCondition[i]) || (front_hightCondition1[i] && front_widthCondition1[i])) {
            results[i] = 1;
        }
        
    }
}
   
    std::vector<std::tuple<int, int, int, int>> back_result = id_card.draw(back_img, back_filtered_boxes);

    // 遍历边界框并处理每个边界框
    //IdCardDirtyArea back_area;
    for (const auto& box : back_result) {
        int left, top, right, bottom;
        std::tie(left, top, right, bottom) = box;
        position.top = top;
        position.left = left;
        position.bottom = bottom;
        position.right = right;
        
        bool back_hightCondition[3];
        bool back_widthCondition[3];
        bool back_hightCondition1[3];
        bool back_widthCondition1[3];
        
        for (int i = 0; i < 3; ++i) 
        {
	    back_hightCondition[i] = (ranges_back_hight[i].max > position.top && ranges_back_hight[i].max < position.bottom) || (ranges_back_hight[i].min > position.top && ranges_back_hight[i].min < position.bottom);

	    back_widthCondition[i] = (ranges_back_width[i].max > position.left && ranges_back_width[i].max < position.right) || (ranges_back_width[i].min > position.left && ranges_back_width[i].min < position.right);
	    
	    back_hightCondition1[i] = (position.bottom > ranges_back_hight[i].min && position.bottom < ranges_back_hight[i].max) || (position.top > ranges_back_hight[i].min && position.top < ranges_back_hight[i].max);

	    back_widthCondition1[i] = (position.right > ranges_back_width[i].min && position.right < ranges_back_width[i].max) || (position.left > ranges_back_width[i].min && position.left < ranges_back_width[i].max);
        }
        

        for (int i = 0; i < 3; ++i) {
        if ((back_hightCondition[i] && back_widthCondition[i]) || (back_hightCondition1[i] && back_widthCondition1[i])) {
            results[i+7] = 1;
        }
        
    }
}

// 打印结果数组以确认
for (int i = 0; i < 12; ++i) {
    if (results[i] == 1) {
        continue;
    }

    results[i] = 0;
}	
	

    return 0;
}
