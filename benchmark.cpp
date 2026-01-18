#include <iostream>
#include <vector>
#include <chrono> // C++ 的计时库
#include <cmath>
#include "yolo_model.h"

#define TIC(name) auto start_##name = std::chrono::high_resolution_clock::now()
#define TOC(name) auto end_##name = std::chrono::high_resolution_clock::now(); \
                  std::cout << #name << " cost: " \
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count() \
                  << " ms" << std::endl

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "🔥 HPC Benchmark: Baseline Test 🔥" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. 准备数据 (制造一个巨大的 Tensor)
    // 形状 [1, 256, 128, 128] -> 约 420 万个浮点数
    // 数据量约 16MB，足够让 CPU 跑一会了
    int N = 1;
    int C = 256;
    int H = 128;
    int W = 128;
    
    std::cout << "Initializing Tensors (" << N*C*H*W << " elements)..." << std::endl;
    Tensor t1({N, C, H, W});
    t1.fill(1.5f); // 填满数据
    
    Tensor t2({N, C, H, W});
    t2.fill(2.5f); // 填满数据

    // 2. 测试 SiLU (计算密集型：涉及 exp 指数运算)
    // 循环跑 50 次，放大时间差异
    std::cout << "\n[Test 1] SiLU Activation (50 loops)..." << std::endl;
    TIC(SiLU);
    for (int i = 0; i < 50; i++) {
        // 这里的 SiLU 还是你原来写的那个朴素版本
        // 后面我们会去 Tensor 类里用 OpenMP/AVX 修改它
        Tensor t3 = t1.SiLU(); 
    }
    TOC(SiLU);

    // 3. 测试 Add (访存密集型：纯加法)
    // 循环跑 100 次
    std::cout << "\n[Test 2] Element-wise Add (100 loops)..." << std::endl;
    TIC(Add);
    for (int i = 0; i < 100; i++) {
        Tensor t4 = t1.Add(t2);
    }
    TOC(Add);

    std::cout << "\nBenchmark Done." << std::endl;
    return 0;
}
