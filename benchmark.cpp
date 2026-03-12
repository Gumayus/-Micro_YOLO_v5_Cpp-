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
