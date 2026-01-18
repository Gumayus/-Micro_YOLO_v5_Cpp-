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

std::vector<Detection> YoloV5Detector::detect(cv::Mat& frame,bool use_quant) {
    // 1. 预处理
    cv::Mat input = formatToSquare(frame);
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(), true, false);

     if (use_quant) {
            std::cout << "⚠️ Warning: Simulating Int8 Quantization loss..." << std::endl;
            
            // 1. 把 cv::Mat 里的数据搬运到你的 Tensor 里
            int N=1, C=3, H=INPUT_H, W=INPUT_W;
            Tensor t_input({N, C, H, W});
            // OpenCV blob 是连续内存，直接拷贝
            memcpy(t_input.data.data(), blob.ptr<float>(), t_input.data.size() * sizeof(float));

            // 2. 执行量化破坏！(Float -> Int8 -> Float)
            Tensor t_quant = t_input.FakeQuantizeInt8();

            // 3. 把破坏后的数据搬回 cv::Mat
            memcpy(blob.ptr<float>(), t_quant.data.data(), t_quant.data.size() * sizeof(float));
        }
    // 2. 推理
      net.setInput(blob);
      std::vector<cv::Mat> outputs;
      net.forward(outputs, net.getUnconnectedOutLayersNames());

    // 3. 后处理 (解析 25200 行数据)
    // outputs[0] 是 [1, 25200, 85]
    float* data = (float*)outputs[0].data;

    // 也就是 rows = 25200
    int rows = outputs[0].size[1];
    // dimensions = 85
    int dimensions = outputs[0].size[2];

    // 还原系数 (因为我们把原图缩放到了 640)
    float x_factor = input.cols / INPUT_W;
    float y_factor = input.rows / INPUT_H;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // --- 核心循环 ---
    for (int i = 0; i < rows; ++i) {
        // 获取当前行的指针
        // [x, y, w, h, obj_conf, class0, class1, ...]
        float* row_ptr = data + i * dimensions;

        // 提取置信度 (第5个数)
        float confidence = row_ptr[4];

        if (confidence >= SCORE_THRES) {
            // 找最大类别的分数
            float* classes_scores = row_ptr + 5;

            // OpenCV 自带找最大值函数 (比手写快)
            cv::Mat scores(1, 80, CV_32FC1, classes_scores);
            cv::Point class_id_point;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            // 最终分数
            if (max_class_score > SCORE_THRES) {
                // 这里的 x, y, w, h 已经是像素值了 (相对于 640x640)
                float x = row_ptr[0];
                float y = row_ptr[1];
                float w = row_ptr[2];
                float h = row_ptr[3];

                // 还原回原图尺寸
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
                class_ids.push_back(class_id_point.x);
            }
        }
    }

    // 4. NMS (非极大值抑制)
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRES, NMS_THRES, nms_result);

    std::vector<Detection> output;
    for (int idx : nms_result) {
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    return output;
}



