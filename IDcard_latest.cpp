#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <array>
#include <card.h>

std::string card_direction ;
std::string model_path;
std::string img_path;
std::vector<std::string> CLASSES = { "occlusion" };

std::vector<int> nms(cv::Mat& dets, float thresh) {
    std::vector<int> keep;

    std::vector<float> x1(dets.rows), y1(dets.rows), x2(dets.rows), y2(dets.rows);
    std::vector<float> areas(dets.rows), scores(dets.rows);

    for (int i = 0; i < dets.rows; i++) {
        x1[i] = dets.at<float>(i, 0);
        y1[i] = dets.at<float>(i, 1);
        x2[i] = dets.at<float>(i, 2);
        y2[i] = dets.at<float>(i, 3);
        areas[i] = (y2[i] - y1[i] + 1) * (x2[i] - x1[i] + 1);
        scores[i] = dets.at<float>(i, 4);
    }

    std::vector<int> indices(dets.rows);
    for (int i = 0; i < dets.rows; i++) {
        indices[i] = i;
    }

    while (!indices.empty()) {
        int i = indices[0];
        keep.push_back(i);

        std::vector<int> new_indices;
        for (size_t j = 1; j < indices.size(); j++) {
            int k = indices[j];

            float xx1 = std::max(x1[i], x1[k]);
            float yy1 = std::max(y1[i], y1[k]);
            float xx2 = std::min(x2[i], x2[k]);
            float yy2 = std::min(y2[i], y2[k]);

            float w = std::max(0.0f, xx2 - xx1 + 1);
            float h = std::max(0.0f, yy2 - yy1 + 1);

            float overlap = w * h;
            float iou = overlap / (areas[i] + areas[k] - overlap);

            if (iou <= thresh) {
                new_indices.push_back(k);
            }
        }

        indices = new_indices;
    }

    return keep;
}

cv::Mat xywh2xyxy(cv::Mat x) {
    cv::Mat y = x.clone();

    for (int i = 0; i < y.rows; i++) {
        y.at<float>(i, 0) = x.at<float>(i, 0) - x.at<float>(i, 2) / 2;
        y.at<float>(i, 1) = x.at<float>(i, 1) - x.at<float>(i, 3) / 2;
        y.at<float>(i, 2) = x.at<float>(i, 0) + x.at<float>(i, 2) / 2;
        y.at<float>(i, 3) = x.at<float>(i, 1) + x.at<float>(i, 3) / 2;
    }

    return y;
}

cv::Mat filter_box(cv::Mat org_box, float conf_thres, float iou_thres) {
    cv::Mat output;

    cv::Mat box_squeezed = org_box.reshape(1, org_box.total() / 6); 

   
    cv::Mat conf = (box_squeezed.col(4) > conf_thres);

  
    cv::Mat box;
    for (int i = 0; i < conf.rows; i++) {
        if (conf.at<uchar>(i) != 0) {
            cv::Mat row = box_squeezed.row(i);
            box.push_back(row);
        }
    }

    std::vector<int> cls;
    for (int i = 0; i < box.rows; i++) {
        cv::Mat cls_cinf = box.row(i).colRange(5, box.cols);
        int max_idx;
        cv::minMaxIdx(cls_cinf, nullptr, nullptr, nullptr, &max_idx);
        cls.push_back(max_idx);
    }

    std::set<int> all_cls(cls.begin(), cls.end());

    std::vector<cv::Rect> output_boxes;

    for (int curr_cls : all_cls) {
        std::vector<cv::Rect> curr_cls_box;
        for (int j = 0; j < cls.size(); j++) {
            if (cls[j] == curr_cls) {
                box.at<float>(j, 5) = curr_cls;
                cv::Rect rect(box.at<float>(j, 0), box.at<float>(j, 1), box.at<float>(j, 2), box.at<float>(j, 3));
                curr_cls_box.push_back(rect);
            }
        }

        cv::Mat curr_cls_box_mat(curr_cls_box.size(), 4, CV_32F);
        for (int k = 0; k < curr_cls_box.size(); k++) {
            curr_cls_box_mat.at<float>(k, 0) = curr_cls_box[k].x;
            curr_cls_box_mat.at<float>(k, 1) = curr_cls_box[k].y;
            curr_cls_box_mat.at<float>(k, 2) = curr_cls_box[k].width;
            curr_cls_box_mat.at<float>(k, 3) = curr_cls_box[k].height;
        }

        cv::Mat curr_out_box = xywh2xyxy(curr_cls_box_mat);

        std::vector<int> curr_keep = nms(curr_out_box, iou_thres);

        for (int k : curr_keep) {
            output_boxes.push_back(curr_cls_box[k]);
        }

        for (const auto& rect : output_boxes) {
            std::cout << "Position: (" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << ")" << std::endl;
        }
    }

    output.create(output_boxes.size(), 6, CV_32F);
    for (int i = 0; i < output_boxes.size(); i++) {
        output.at<float>(i, 0) = output_boxes[i].x - output_boxes[i].width / 2;
        output.at<float>(i, 1) = output_boxes[i].y - output_boxes[i].height / 2;
        output.at<float>(i, 2) = output_boxes[i].x + output_boxes[i].width / 2;
        output.at<float>(i, 3) = output_boxes[i].y + output_boxes[i].height / 2;
        output.at<float>(i, 4) = box.at<float>(i, 4);
        output.at<float>(i, 5) = box.at<float>(i, 5);
    }
    return output;
}

void draw(cv::Mat& image, cv::Mat& box_data) {
    for (int i = 0; i < box_data.rows; i++) {
        int left = box_data.at<float>(i, 0);
        int top = box_data.at<float>(i, 1);
        int right = box_data.at<float>(i, 2);
        int bottom = box_data.at<float>(i, 3);
        float score = box_data.at<float>(i, 4);
        int cls = static_cast<int>(box_data.at<float>(i, 5));

        // std::cout << "class: " << CLASSES[cls] << ", score: " << score << std::endl;
        // std::cout << "box coordinate left,top,right,down: [" << left << ", " << top << ", " << right << ", " << bottom << "]" << std::endl;

        // cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 0, 0), 2);
        // cv::putText(image, CLASSES[cls] + " " + std::to_string(score), cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        return left, top, right, bottom;
    }
}

int main() 
{
    cv::dnn::Net net = cv::dnn::readNetFromONNX("model_path");
    cv::Mat img = cv::imread("img_path");
    cv::resize(img, img, cv::Size(640, 640));
    cv::Mat blob = cv::dnn::blobFromImage(img, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    cv::Mat detection = net.forward();
    cv::Mat filtered_boxes = filter_box(detection, 0.25, 0.45);
    left, top, right, bottom = draw(img, filtered_boxes);

    // cv::imshow("Detection Result", img);
    // cv::waitKey(0);
    return 0;
}