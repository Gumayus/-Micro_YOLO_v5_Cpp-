#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "yolo_model_v5.h" // 你的 YOLO 头文件
#include "MicroKalman.h"   // 你的 Kalman 头文件

// 辅助：获取当前时间戳 (ms) 用于计算 FPS
double get_time_ms() {
    return (double)cv::getTickCount() * 1000.0 / cv::getTickFrequency();
}

int main() {
    std::cout << "=== YOLOv5 + Kalman Tracking (Image Loop Mode) ===" << std::endl;

    // 1. 初始化 YOLO
    // 确保 yolov5n_sim.onnx 在当前目录 (build_cpp)
    std::cout << "[Init] Loading YOLOv5 model..." << std::endl;
    YoloV5Detector detector("yolov5n_sim.onnx", false); // false = CPU FP32

    // 2. 初始化 Kalman
    // 假设循环大约 30ms 一次，dt = 0.033
    std::cout << "[Init] Initializing MicroKalman..." << std::endl;
    KalmanFilter kf(0.033f);

    // 3. 读取静态图片 (替代视频流)
    // 确保 bus.jpg 在当前目录
    std::string img_path = "bus.jpg";
    cv::Mat source_img = cv::imread(img_path);

    if (source_img.empty()) {
        std::cerr << "❌ Error: Could not load " << img_path << std::endl;
        std::cerr << "   请把 bus.jpg 复制到 build_cpp 文件夹下！" << std::endl;
        return -1;
    }
    std::cout << "✅ Image loaded: " << source_img.cols << "x" << source_img.rows << std::endl;

    bool is_initialized = false; // 卡尔曼是否锁定目标
    int frame_count = 0;

    std::cout << "🚀 Loop started! Press 'ESC' to exit." << std::endl;

    while (true) {
        double t_start = get_time_ms();

        // --- A. 模拟视频流 ---
        // 每次 clone 一份新的，防止画图画花了原图
        cv::Mat frame = source_img.clone(); 
        
        // --- B. YOLO 检测 ---
        // 使用 FP32 模式 (use_quant=false) 保证最稳
        std::vector<Detection> results = detector.detect(frame, false);

        // --- C. 寻找目标 (逻辑：只追置信度最高的 Person) ---
        Detection best_target;
        bool found_target = false;

        for (const auto& det : results) {
            // Class 0 是 Person
            if (det.class_id == 0) {
                if (!found_target || det.confidence > best_target.confidence) {
                    best_target = det;
                    found_target = true;
                }
            }
        }

        // --- D. 卡尔曼滤波闭环 ---
        
        // D1. 预测 (Predict) - 必须每帧都做！
        // 如果已经初始化了，就根据惯性猜一下位置
        if (is_initialized) {
            kf.predict();
        }

        // D2. 更新 (Update) - 只有看到目标才做！
        if (found_target) {
            // 提取中心点坐标
            float cx = best_target.box.x + best_target.box.width / 2.0f;
            float cy = best_target.box.y + best_target.box.height / 2.0f;

            // 构造观测向量 z [2x1]
            Tensor z({2, 1});
            z.data[0] = cx;
            z.data[1] = cy;

            if (!is_initialized) {
                // 冷启动：第一次看到人，瞬移过去，不进行滤波
                // 简单粗暴的方法：多次 update 让它快速收敛
                // 或者你的 MicroKalman 如果有 reset 接口更好，这里用 update 代替
                for(int i=0; i<5; ++i) kf.update(z); 
                is_initialized = true;
            } else {
                // 正常修正
                kf.update(z);
            }

            // 画 YOLO 框 (绿色)
            cv::rectangle(frame, best_target.box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "YOLO", cv::Point(best_target.box.x, best_target.box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        // --- E. 绘制卡尔曼预测结果 (红色十字) ---
        if (is_initialized) {
            Tensor state = kf.getState(); // [x, y, vx, vy]
            float k_x = state.data[0];
            float k_y = state.data[1];

            // 画一个红色的十字瞄准线
            cv::drawMarker(frame, cv::Point((int)k_x, (int)k_y), 
                           cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 30, 3);
            
            // 显示预测坐标
            std::string info = "KF: (" + std::to_string((int)k_x) + ", " + std::to_string((int)k_y) + ")";
            cv::putText(frame, info, cv::Point((int)k_x + 15, (int)k_y), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        // --- F. 显示 FPS ---
        double t_end = get_time_ms();
        float fps = 1000.0f / (t_end - t_start);
        cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(20, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        cv::imshow("AIoT Tracking System", frame);

        // 30ms 延时 (模拟 30FPS)
        if (cv::waitKey(30) == 27) break; 
        
        frame_count++;
    }

    return 0;
}