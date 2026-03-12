#pragma one
#include <iostream>
#include <memory>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <iomanip>
#include <string>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <memory>
#include "yolo_model.h"

class KalmanFilter
{
public:
    // dt : 时间间隔
    KalmanFilter(float dt = 0.033f);
    KalmanFilter(KalmanFilter &&) = default;
    KalmanFilter &operator=(KalmanFilter &&) = default;
    KalmanFilter(const KalmanFilter &) = delete;
    KalmanFilter &operator=(const KalmanFilter &) = delete;

    // 状态预测
    void predict();

    // 状态更新 观测值z [2x1]
    void update(const Tensor &z);

    // 获取当前状态估计值 [x,y,vx,vy]
    Tensor getState() const { return *X; }

    // 强行设置当前状态
    //  init_x, init_y: 初始坐标
    //  init_vx, init_vy: 初始速度 (通常设为0)

    void set_state(float init_x, float init_y, float init_vx = 0, float init_vy = 0);

    // 七大矩阵
    std::unique_ptr<Tensor> X; // 状态向量 [4x1]
    std::unique_ptr<Tensor> P; // 协方差矩阵 [4x4],描述状态估计的不确定性
    std::unique_ptr<Tensor> F; // 状态转移矩阵 [4x4],描述系统状态如何随时间变化
    std::unique_ptr<Tensor> H; // 观测矩阵 [2x4],描述如何从状态空间映射到观测空间
    std::unique_ptr<Tensor> Q; // 过程噪声协方差矩阵 [4x4],描述系统过程中的不确定性
    std::unique_ptr<Tensor> R; // 测量噪声协方差矩阵 [2x2],描述观测中的不确定性
    std::unique_ptr<Tensor> I; // 单位矩阵 [4x4]

    // 初始化所有矩阵
    void init_matrices(float dt);
};
