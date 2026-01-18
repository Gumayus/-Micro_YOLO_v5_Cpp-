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



// ... 你的 YoloV5Detector 实现代码 ...

// ==========================================
// 主函数：程序的入口
// ==========================================
// int main() {
//     // 1. 初始化检测器
//     // "yolov5n.onnx" 就是你刚才放进去的那个文件
//     // false 代表使用 CPU 模式 (先跑通 CPU 再说)
//     std::cout << "--- Initializing YOLOv5... ---" << std::endl;
//     YoloV5Detector detector("yolov5n_sim.onnx", false);

//     // 2. 读取测试图片
//     // "bus.jpg" 必须和 .onnx 文件在同一个文件夹
//     std::string imgPath = "bus.jpg";
//     cv::Mat img = cv::imread(imgPath);

//     // 检查图片是否读取成功
//     if (img.empty()) {
//         std::cerr << "❌ Error: Image not found at " << imgPath << std::endl;
//         std::cerr << "请确保 bus.jpg 已经放在了项目根目录下！" << std::endl;
//         return -1;
//     }
//     std::cout << "✅ Image loaded: " << img.cols << "x" << img.rows << std::endl;

//     // 3. 执行推理
//     std::cout << "🚀 Running inference..." << std::endl;
//     std::vector<Detection> results = detector.detect(img);
//     std::cout << "🔍 Found " << results.size() << " objects." << std::endl;

//     // 4. 画图展示结果
//     // 准备 COCO 80 类别的名字 (为了显示好看)
//     std::vector<std::string> classNames = {
//         "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//         "hair drier", "toothbrush"
//     };

//     for (const auto& res : results) {
//         // 画框
//         cv::rectangle(img, res.box, cv::Scalar(0, 255, 0), 2); // 绿色框

//         // 准备标签文字
//         std::string label = "Unknown";
//         if (res.class_id >= 0 && res.class_id < classNames.size()) {
//             label = classNames[res.class_id];
//         }
//         label += ": " + std::to_string(res.confidence).substr(0, 4);

//         // 画文字背景
//         int baseLine;
//         cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//         cv::rectangle(img,
//             cv::Point(res.box.x, res.box.y - labelSize.height - 5),
//             cv::Point(res.box.x + labelSize.width, res.box.y),
//             cv::Scalar(0, 255, 0), cv::FILLED);

//         // 写文字
//         cv::putText(img, label, cv::Point(res.box.x, res.box.y - 5),
//             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

//         // 打印到控制台
//         std::cout << "   -> " << label << " @ [" << res.box.x << "," << res.box.y << "]" << std::endl;
//     }

//     // 5. 弹窗显示 (按任意键退出)
//     cv::imshow("YOLOv5 C++ Result", img);
//     cv::waitKey(0);

//     return 0;
// }