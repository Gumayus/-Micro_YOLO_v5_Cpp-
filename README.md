# Micro-YOLOv5 C++ Inference Engine 🚀

> **硬核驱动，死磕底层。** 这是一个从零手写的 YOLOv5 推理引擎，不依赖任何第三方深度学习框架，专为嵌入式边缘侧（AIoT）设计。

## 🌟 核心特性 (Features)
*   **Zero Dependency**: 纯 C++ 实现，算子级手搓。
*   **HPC Optimization**: 
    *   **SIMD 加速**: 利用 AVX2 指令集（Intrinsics）重写算子。
    *   **Cache Awareness**: 优化 i-k-j 循环顺序，性能提升 30 倍。
*   **Quantization Engine**: 自主实现 **Int8 纯整数推理内核**，支持动态校准。

## 📉 量化精度分析 (Quantization Analysis)
对 Stem Layer 进行了 FP32 与 Int8 的精度对齐实验。

**实验结果：**
*   **MSE (均方误差)**: `0.006851`
*   **结论**: 误差呈均匀随机分布，完美保留了特征语义，满足边缘端部署精度要求。

![Quantization Comparison](assets/quant_analysis.png)

## 🛠️ 构建与运行 (Build & Run)
```bash
mkdir build && cd build
cmake ..
make -j8
./yolo_run