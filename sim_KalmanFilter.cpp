#include "MicroKalman.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    std::cout << "=== Kalman Filter Simulation Start ===" << std::endl;

    // 1. 初始化模拟器参数
    float dt = 0.1f; // 假设 100ms 一帧 (10Hz)
    KalmanFilter kf(dt);

    // 真实状态 [x, y, vx, vy]
    // 初始位置 (0,0), 速度 (10, 5)
    float true_x = 0.0f;
    float true_y = 0.0f;
    float true_vx = 10.0f;
    float true_vy = 5.0f;

    // 随机数生成器 (模拟 YOLO 的误差)
    std::default_random_engine generator;
    std::normal_distribution<float> noise(0.0f, 2.0f); // 均值0，标准差2.0像素

    // 2. 开始仿真循环 (跑 20 帧)
    std::cout << "Frame |   True(x,y)   | Measured(x,y) | Estimated(x,y) |  Error(Meas) | Error(Est)" << std::endl;
    std::cout << "-------------------------------------------------------------------------------------" << std::endl;

    for (int i = 0; i < 20; ++i) {
        // --- A. 物理世界演化 (上帝视角) ---
        true_x += true_vx * dt;
        true_y += true_vy * dt;

        // --- B. 生成带噪声的观测值 (模拟 YOLO) ---
        float meas_x = true_x + noise(generator);
        float meas_y = true_y + noise(generator);

        Tensor z({2, 1});
        z.data[0] = meas_x;
        z.data[1] = meas_y;

        // --- C. 卡尔曼滤波 (你的核心代码) ---
        // 1. 预测 (先验)
        kf.predict();
        
        // 2. 更新 (后验)
        kf.update(z);

        // 3. 获取结果
        Tensor state = kf.getState();
        float est_x = state.data[0];
        float est_y = state.data[1];

        // --- D. 打印对比 ---
        float err_meas = std::abs(meas_x - true_x) + std::abs(meas_y - true_y);
        float err_est  = std::abs(est_x - true_x)  + std::abs(est_y - true_y);

        std::cout << std::setw(5) << i << " | "
                  << "[" << std::setw(4) << (int)true_x << "," << std::setw(4) << (int)true_y << "] | "
                  << "[" << std::setw(4) << (int)meas_x << "," << std::setw(4) << (int)meas_y << "] | "
                  << "[" << std::setw(4) << (int)est_x << "," << std::setw(4) << (int)est_y << "] | "
                  << std::setw(8) << std::fixed << std::setprecision(2) << err_meas << " | "
                  << std::setw(8) << std::fixed << std::setprecision(2) << err_est 
                  << std::endl;
    }

    return 0;
}