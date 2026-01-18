#include "yolo_model_v5.h"
#include "yolo_model.h"
#include <iostream>
#include <string>
#include <vector>

// 辅助函数：画框和文字
void draw_box(cv::Mat& img, const Detection& res, const std::string& prefix, cv::Scalar color, int thickness) {
    // 准备 COCO 80 类别的名字
    static const std::vector<std::string> classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    // 1. 画框
    cv::rectangle(img, res.box, color, thickness);

    // 2. 准备标签文字
    std::string label = "Unknown";
    if (res.class_id >= 0 && res.class_id < classNames.size()) {
        label = classNames[res.class_id];
    }
    // 加上前缀 (FP32/Int8) 和置信度
    label = prefix + label + ": " + std::to_string(res.confidence).substr(0, 4);

    // 3. 画文字背景 (为了看清字)
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    
    // 稍微错开一点位置，防止文字重叠
    int text_y = res.box.y - 5;
    if (prefix == "[Int8] ") text_y -= 15; // 如果是 Int8，文字往上提一点

    cv::putText(img, label, cv::Point(res.box.x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    
    // 控制台打印
    std::cout << "   -> " << label << " @ [" << res.box.x << "," << res.box.y << "]" << std::endl;
}

int main() {
    // 1. 初始化
    std::cout << "--- Initializing YOLOv5... ---" << std::endl;
    // 使用简化过的 FP32 模型
    YoloV5Detector detector("yolov5n_sim.onnx", false);

    // 2. 读取图片
    std::string imgPath = "bus.jpg";
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        std::cerr << "❌ Error: Image not found!" << std::endl;
        return -1;
    }
    std::cout << "✅ Image loaded: " << img.cols << "x" << img.rows << std::endl;

    // ---------------------------------------------------------
    // 3. Round 1: FP32 推理 (基准)
    // ---------------------------------------------------------
    std::cout << "\n🚀 [Round 1] Running FP32 Inference (Baseline)..." << std::endl;
    // use_quant = false
    std::vector<Detection> results_fp32 = detector.detect(img, false);
    std::cout << "🔍 FP32 Found " << results_fp32.size() << " objects." << std::endl;

    // ---------------------------------------------------------
    // 4. Round 2: Int8 模拟推理 (破坏性测试)
    // ---------------------------------------------------------
    std::cout << "\n🔨 [Round 2] Running Int8 Simulation (Quantization)..." << std::endl;
    // use_quant = true -> 这一步会把你刚手写的 FakeQuantizeInt8 用起来！
    std::vector<Detection> results_int8 = detector.detect(img, true);
    std::cout << "🔍 Int8 Found " << results_int8.size() << " objects." << std::endl;

    // ---------------------------------------------------------
    // 5. 绘图对比
    // ---------------------------------------------------------
    std::cout << "\n🎨 Drawing results..." << std::endl;

    // 先画 FP32 (绿色，线宽 2)
    for (const auto& res : results_fp32) {
        draw_box(img, res, "[FP32] ", cv::Scalar(0, 255, 0), 2);
    }

    // 后画 Int8 (红色，线宽 1) -> 这样红色会叠加在绿色上面，方便看偏差
    for (const auto& res : results_int8) {
        draw_box(img, res, "[Int8] ", cv::Scalar(0, 0, 255), 1);
    }

    // 6. 显示结果
    cv::imshow("YOLOv5: FP32(Green) vs Int8(Red)", img);
    cv::waitKey(0);

    return 0;
}