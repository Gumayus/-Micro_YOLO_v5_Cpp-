#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>

//定义缓存区]

const int N =1024;

//定义计数时间宏
#define TIC(name) auto start_##name = std::chrono::high_resolution_clock::now()
#define TOC(name) auto end_##name = std::chrono::high_resolution_clock::now(); \
                  std::cout << #name << " cost: " \
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count() \
                  << " ms" << std::endl


void matmul_naive(const float*A,const float*B,float*C)
{
    std::fill(C,C+N*N,0.0f);

    for (int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            float sum =0.0f;
            for (int k=0;k<N;k++)
            {
                sum+=A[i*N+k] *B[k*N +j];

            }

            C[i*N+j]=sum;
        }
    }
}

void matmul_optimized(const float* A, const float* B, float* C) {
    std::fill(C, C + N*N, 0.0f);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) { 
            // Temporal Locality: A[i][k] 在内层循环保持不变，放入寄存器
            float r = A[i*N + k];     
            
            for (int j = 0; j < N; j++) { 
                // Spatial Locality: C[i][j] 和 B[k][j] 都是 j++
                // 完美命中 Cache Line！
                C[i*N + j] += r * B[k*N + j]; 
            }
        }
    }
}


int main() {
    std::cout << "--- MatMul Benchmark (N=" << N << ") ---" << std::endl;

    // 分配内存
    std::vector<float> A(N*N, 1.0f);
    std::vector<float> B(N*N, 2.0f);
    std::vector<float> C1(N*N, 0.0f);
    std::vector<float> C2(N*N, 0.0f);

    // 预热 Cache
    matmul_optimized(A.data(), B.data(), C2.data());

    // 测速 1: Naive (i-j-k)
    std::cout << "Running Naive (i-j-k)..." << std::endl;
    TIC(Naive);
    matmul_naive(A.data(), B.data(), C1.data());
    TOC(Naive);

    // 测速 2: Optimized (i-k-j)
    std::cout << "Running Optimized (i-k-j)..." << std::endl;
    TIC(Optimized);
    matmul_optimized(A.data(), B.data(), C2.data());
    TOC(Optimized);

    // 校验结果 (防止优化过度导致结果错误)
    float err = 0;
    for(int i=0; i<N*N; i++) err += std::abs(C1[i] - C2[i]);
    std::cout << "Error check: " << err << std::endl;

    return 0;
}