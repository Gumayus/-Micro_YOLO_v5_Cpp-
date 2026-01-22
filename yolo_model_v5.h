#pragma once
#include "yolo_model.h"
#include <iostream>

YoloV5Detector::YoloV5Detector(const std::string& modelPath, bool isCuda) {
    net = cv::dnn::readNetFromONNX(modelPath);
    if (isCuda) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat YoloV5Detector::formatToSquare(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<Detection> YoloV5Detector::detect(cv::Mat& frame, bool use_quant) {
    // 1. 图像预处理 (Letterbox)
    cv::Mat input = formatToSquare(frame);
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(), true, false);
    
    // 2. 推理
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // 3. 后处理准备
    float* data = (float*)outputs[0].data;
    int rows = outputs[0].size[1]; // 25200
    int dimensions = outputs[0].size[2]; // 85

    float x_factor = input.cols / INPUT_W;
    float y_factor = input.rows / INPUT_H;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // 4. 遍历与初步筛选
    for (int i = 0; i < rows; ++i) {
        float* row_ptr = data + i * dimensions;
        float confidence = row_ptr[4];

        if (confidence >= SCORE_THRES) {
            float* classes_scores = row_ptr + 5;
            cv::Mat scores(1, 80, CV_32FC1, classes_scores);
            cv::Point class_id_point;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            if (max_class_score > SCORE_THRES) {
                float cx = row_ptr[0];
                float cy = row_ptr[1];
                float w = row_ptr[2];
                float h = row_ptr[3];

                // 还原回原图像素坐标
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
                class_ids.push_back(class_id_point.x);
            }
        }
    }

    // 5. NMS
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRES, NMS_THRES, nms_result);

    // 6. 核心改造：封装并同步数据
    std::vector<Detection> output;
    for (int idx : nms_result) {
        Detection res;
        res.class_id = class_ids[idx];
        res.confidence = confidences[idx];
        res.box = boxes[idx];

        // 🔥【关键动作】调用同步函数
        // 这一步把 cv::Rect 的整数坐标，转化成卡尔曼要用的浮点中心点
        res.sync();

        output.push_back(res);
    }

    return output;
}




