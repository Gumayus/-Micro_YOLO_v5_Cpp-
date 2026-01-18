#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
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
#include <omp.h>
#include <immintrin.h>

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class YoloV5Detector
{
public:
    // 构造函数
    YoloV5Detector(const std::string &modelPath, bool isCuda = false);
    // 检测函数
    std::vector<Detection> detect(cv::Mat &frame, bool use_quant = false);

private:
    // 参数
    const float INPUT_W = 640.0;
    const float INPUT_H = 640.0;
    const float SCORE_THRES = 0.15f; // 置信度阈值
    const float NMS_THRES = 0.45f;   // NMS 阈值

    cv::dnn::Net net;

    // 预处理Letterbox (保持宽高比例)
    cv::Mat formatToSquare(const cv::Mat &source);
};

inline float random_normal()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dis(0.0f, 1.0f); // 标准正态分布，均值为0，标准差为1
    return dis(gen);
}

class Tensor
{
public:
    std::vector<float> data;
    std::vector<int> shape;   // 张量形状
    std::vector<int> strides; // 步长

    void compute_strides()
    {
        strides.resize(shape.size()); // 将步长数组大小设置为与shape数组相同

        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride; // 设定最后一维的步长为1
            stride *= shape[i];  // 去除最后一维后，前一个维度的步长等于当前维度的元素个数乘以前一维的步长
        }
    }

    Tensor() {}

    static Tensor randn(const std::vector<int> &shape)
    {
        Tensor t{shape};
        for (size_t i = 0; i < t.data.size(); i++)
        {
            t.data[i] = random_normal();
        }

        return t;
    }

    Tensor(std::vector<int> shape) : shape(shape)
    {
        int size = 1;
        for (int s : shape)
            size *= s;

        data.resize(size, 0.0f); // 初始化数据，每个元素为0

        compute_strides();
    }

    void fill(float value)
    {
        std::fill(data.begin(), data.end(), value); // 将data中的所有元素设置为value
    }

    Tensor operator+(const Tensor &other) const
    {
        if (this->shape == other.shape)
        {
            Tensor result(this->shape);

            for (size_t i = 0; i < data.size(); ++i)
            {
                result.data[i] = data[i] + other.data[i];
            }

            return result;
        }

        else if (this->shape.size() == 2 && other.shape.size() == 1 && other.shape[0] == this->shape[1])
        {
            Tensor result(this->shape);
            int rows = this->shape[0];
            int cols = this->shape[1];

            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    int self_ix = i * strides[0] + j * strides[1];
                    int other_ix = j;

                    result.data[self_ix] = this->data[self_ix] + other.data[other_ix];
                }
            }

            return result;
        }

        else
        {
            std::cerr << "Error: Unsupported broadcasting shapes: "
                      << "[" << shape[0] << ", ...]" << " + "
                      << "[" << other.shape[0] << ", ...]" << std::endl;
            exit(1);
        }
    }

    Tensor operator-(const Tensor &other) const
    {
        if (this->shape == other.shape)
        {
            Tensor result(this->shape);

            for (size_t i = 0; i < data.size(); ++i)
            {
                result.data[i] = data[i] - other.data[i];
            }

            return result;
        }

        else if (this->shape.size() == 2 && other.shape.size() == 1 && other.shape.size() == this->shape[1])
        {
            Tensor result(this->shape);
            int rows = this->shape[0];
            int cols = this->shape[1];

            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    int self_ix = i * strides[0] + j * strides[1];
                    int other_ix = j;

                    result.data[self_ix] = this->data[self_ix] - other.data[other_ix];
                }
            }

            return result;
        }

        else
        {
            std::cerr << "Error: Unsupported broadcasting shapes: "
                      << "[" << shape[0] << ", ...]" << " + "
                      << "[" << other.shape[0] << ", ...]" << std::endl;
            exit(1);
        }
    }

    Tensor operator*(float LR) const
    {
        Tensor result(this->shape);

        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = data[i] * LR;
        }

        return result;
    }

    Tensor operator/(float scalar) const
    {
        Tensor result(this->shape);
        float inv = 1.0f / scalar;
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = data[i] * inv;
        }

        return result;
    }

    Tensor operator/(const Tensor &other) const
    {
        int ndim = this->shape.size();

        if (this->shape == other.shape)
        {
            Tensor result(this->shape);
            for (size_t i = 0; i < data.size(); ++i)
            {
                result.data[i] = data[i] / other.data[i];
            }
            return result;
        }
        // 2. 2D 广播: [M, N] / [M, 1]
        else if (ndim == 2 && other.shape.size() == 2 &&
                 other.shape[1] == 1 && this->shape[0] == other.shape[0])
        {
            Tensor result(this->shape);
            int rows = shape[0];
            int cols = shape[1];
            for (int i = 0; i < rows; ++i)
            {
                float div = other.at(i, 0); // 取除数的值
                for (int j = 0; j < cols; ++j)
                {
                    result.at(i, j) = this->at(i, j) / div;
                }
            }
            return result;
        }
        // 3. 特殊情况：4D 广播: [B, NH, T, D] / [B, NH, T, 1] (GPT Attention 场景)
        else if (ndim == 4 && other.shape.size() == 4 &&
                 other.shape[3] == 1 && // 最后一维是 1
                 this->shape[0] == other.shape[0] &&
                 this->shape[1] == other.shape[1] &&
                 this->shape[2] == other.shape[2])
        {
            Tensor result(this->shape);
            int B = shape[0];
            int NH = shape[1];
            int T = shape[2];
            int D = shape[3];

            for (int b = 0; b < B; ++b)
            {
                for (int nh = 0; nh < NH; ++nh)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        // 取分母 (最后一维索引0)
                        float div = other.at_4d(b, nh, t, 0);
                        // 将一行的所有元素除以同一个值
                        for (int d = 0; d < D; ++d)
                        {
                            result.at_4d(b, nh, t, d) = this->at_4d(b, nh, t, d) / div;
                        }
                    }
                }
            }
            return result;
        }

        std::cerr << "Error: Unsupported division shape!" << std::endl;
        exit(1);
    }

    Tensor exp() const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = std::exp(data[i]);
        }

        return result;
    }

    float &at(int i, int j)
    {
        return data[i * strides[0] + j * strides[1]];
    }

    const float &at(int i, int j) const
    {
        return data[i * strides[0] + j * strides[1]];
    }

    // 嵌入层
    static Tensor Embedding(const std::vector<int> &input_indices, const Tensor &weight)
    {
        int N = input_indices.size();    // 一次批量查询的字符数
        int embed_dim = weight.shape[1]; // 嵌入层权重维度

        Tensor out({N, embed_dim});

        for (int i = 0; i < N; i++)
        {
            int ix = input_indices[i];

            if (ix < 0 || ix >= weight.shape[0])
            {
                std::cerr << "Embedding index out of bounds!" << std::endl;
                exit(1);
            }

            memcpy(&out.at(i, 0), &weight.at(ix, 0), embed_dim * sizeof(float));
        }

        return out;
    }

    // 矩阵乘法
    Tensor MatMul(const Tensor &other) const
    {
        int ndim = this->shape.size();

        // 1. 维度检查
        if (ndim != other.shape.size())
        {
            std::cerr << "MatMul Error: Ranks must match! " << ndim << " vs " << other.shape.size() << std::endl;
            exit(1);
        }
        if (ndim != 2 && ndim != 4)
        {
            std::cerr << "MatMul Error: Currently only supports 2D or 4D tensors (for GPT)." << std::endl;
            exit(1);
        }

        // 2. 维度提取 (假设形状为 A[..., M, K] @ B[..., K, N])
        // 无论是 2D 还是 4D，最后两维都是矩阵乘法的核心
        int M = this->shape[ndim - 2];
        int K = this->shape[ndim - 1];
        int K_other = other.shape[ndim - 2];
        int N = other.shape[ndim - 1];

        // 3. K 维度必须相等
        if (K != K_other)
        {
            std::cerr << "MatMul Shape Mismatch: "
                      << "A[..." << M << "," << K << "] @ B[..." << K_other << "," << N << "]" << std::endl;
            exit(1);
        }

        // 4. 合并 Batch 维度 (前 N-2 维必须一致)
        // 确定输出形状
        std::vector<int> out_shape = this->shape;
        out_shape[ndim - 1] = N; // 最后一维变成 N

        for (int i = 0; i < ndim - 2; ++i)
        {
            if (this->shape[i] != other.shape[i])
            {
                std::cerr << "MatMul Batch Dimension Mismatch!" << std::endl;
                exit(1);
            }
        }

        // 5. 创建输出张量
        Tensor out(out_shape);

        // =================================================
        //  情况 A: 2D 矩阵乘法 (Linear 层)
        // =================================================
        if (ndim == 2)
        {
            // OpenMP 并行计算 (如果系统支持)
#pragma omp parallel for collapse(2)
            for (int i = 0; i < M; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k)
                    {
                        // 使用 at(i, j) 自动计算 strides
                        sum += this->at(i, k) * other.at(k, j);
                    }
                    out.at(i, j) = sum;
                }
            }
        }
        // =================================================
        //  情况 B: 4D 批量矩阵乘法 (Attention 层)
        //  形状: [Batch, Head, Seq, Dim]
        // =================================================
        else if (ndim == 4)
        {
            int B_dim = shape[0];  // Batch Size
            int NH_dim = shape[1]; // Num Heads

            // 嵌套循环太多？其实前两维是"批量"维度，后两维是"计算"维度
#pragma omp parallel for collapse(2) // 并行化每个 Batch 和 Head
            for (int b = 0; b < B_dim; ++b)
            {
                for (int nh = 0; nh < NH_dim; ++nh)
                {

                    // 接下来对每个 (b, nh) 下的矩阵做乘法
                    // [M, K] @ [K, N] -> [M, N]
                    for (int i = 0; i < M; ++i)
                    {
                        for (int j = 0; j < N; ++j)
                        {
                            float sum = 0.0f;
                            for (int k = 0; k < K; ++k)
                            {
                                // 这里魔法：使用 at_4d 偷看 stride 规律
                                // 即使 K 维有 transpose 转置，at_4d 也能找到正确位置
                                float val_a = this->at_4d(b, nh, i, k);
                                float val_b = other.at_4d(b, nh, k, j);
                                sum += val_a * val_b;
                            }
                            out.at_4d(b, nh, i, j) = sum;
                        }
                    }
                }
            }
        }

        return out;
    }

    // 重排形状
    Tensor view(const std::vector<int> &new_shape) const // 传入你想变成的shape
    {
        int current_size = 1;
        for (int s : this->shape)
            current_size *= s;

        int new_size = 1;
        for (int s : new_shape)
            new_size *= s;

        if (current_size != new_size)
        {
            std::cerr << "Error: Shape mismatch in view! Cannot reshape "
                      << current_size << " elements to " << new_size << std::endl;
            exit(1);
        }

        Tensor out = *this;

        out.shape = new_shape;

        out.compute_strides();

        return out;
    }

    // 激活函数

    Tensor tanh() const
    {
        Tensor out(this->shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            out.data[i] = std::tanh(data[i]);
        }

        return out;
    }

    Tensor sum(int axis = -1) const
    {
        int ndim = shape.size();

        // 1. 处理负数轴 (例如 -1 表示最后一维)
        if (axis < 0)
            axis += ndim;

        // 2. 目前只支持对"最后一维"求和 (Softmax/LayerNorm 场景)
        if (axis != ndim - 1)
        {
            std::cerr << "Sum Error: Currently only supports summing over the LAST dimension!" << std::endl;
            exit(1);
        }

        // 3. 确定输出形状
        // 其他维度保持不变，最后一维变为 1 (KeepDim=True)
        std::vector<int> out_shape = this->shape;
        out_shape[axis] = 1;
        Tensor out(out_shape);

        // =================================================
        //  情况 A: 2D 求和 [rows, cols] -> [rows, 1]
        // =================================================
        if (ndim == 2)
        {
            int rows = shape[0];
            int cols = shape[1];

            for (int i = 0; i < rows; ++i)
            {
                float total = 0.0f;
                for (int j = 0; j < cols; ++j)
                {
                    total += this->at(i, j);
                }
                out.at(i, 0) = total;
            }
        }
        // =================================================
        //  情况 B: 4D 求和 [B, NH, T, D] -> [B, NH, T, 1]
        // =================================================
        else if (ndim == 4)
        {
            int B = shape[0];
            int NH = shape[1];
            int T = shape[2];
            int D = shape[3]; // 最后一维

            // 遍历前三维
            for (int b = 0; b < B; ++b)
            {
                for (int nh = 0; nh < NH; ++nh)
                {
                    for (int t = 0; t < T; ++t)
                    {

                        float total = 0.0f;
                        // 对最后一维累加
                        for (int d = 0; d < D; ++d)
                        {
                            total += this->at_4d(b, nh, t, d);
                        }

                        // 存入结果 (最后一维索引 0)
                        out.at_4d(b, nh, t, 0) = total;
                    }
                }
            }
        }
        else
        {
            std::cerr << "Sum Error: Only 2D and 4D tensors supported." << std::endl;
            exit(1);
        }

        return out;
    }

    // Softmax

    Tensor softmax(int axis = -1) const
    {
        Tensor exps = this->exp();

        Tensor sum = exps.sum(axis);

        return exps / sum;
    }

    // GPT的激活函数
    Tensor GELU() const
    {
        Tensor out(this->shape);
        const float k0 = 0.7978845608f;
        const float k1 = 0.044715f;
        for (size_t i = 0; i < data.size(); ++i)
        {
            float val = data[i];
            float x = k0 * (val + k1 * (val * val * val));
            out.data[i] = 0.5f * data[i] * (1.0f + std::tanh(x));
        }
        return out;
    }

    // 层归一化 对最后一维进行归一化
    Tensor LayerNorm(const Tensor &gamma, const Tensor &beta) const // gamma 缩放参数  beta 偏移参数
    {
        Tensor out(this->shape);

        int D = this->shape.back();
        int N = this->data.size() / D;

        const float eps = 1e-5f;

        for (int i = 0; i < N; ++i)
        {
            int offset = i * D; // 找到每一行的起始位置

            // 计算均值 mean
            float sum = 0.0f;
            for (int j = 0; j < D; ++j)
            {
                sum += data[offset + j];
            }

            float mean = sum / D;

            // 计算方差
            float sum_sq_diff = 0.0f;
            for (int j = 0; j < D; ++j)
            {
                float diff = data[offset + j] - mean;
                sum_sq_diff += diff * diff;
            }

            float variance = sum_sq_diff / D;

            // 计算标准化系数

            float inv_std = 1.0f / std::sqrt(variance + eps); // sqrt 平方根

            // 归一化

            for (int j = 0; j < D; ++j)
            {
                float n = (data[offset + j] - mean) * inv_std;

                out.data[offset + j] = n * gamma.data[j] + beta.data[j];
            }
        }

        return out;
    }

    Tensor Transpose(int dim0, int dim1) const // 需要交换的两个维度
    {
        Tensor out = *this;
        int ndim = out.shape.size();
        if (dim0 < 0)
            dim0 += ndim;

        if (dim1 < 0)
            dim1 += ndim;

        if (dim0 >= ndim || dim1 >= ndim) // 边界检查
        {
            std::cerr << "Error: Transpose dim out of bounds!" << std::endl;
            exit(1);
        }

        std::swap(out.shape[dim0], out.shape[dim1]); // 交换形状

        std::swap(out.strides[dim0], out.strides[dim1]); // 交换步长

        return out;
    }

    float &at_4d(int i0, int i1, int i2, int i3)
    {
        int offset = i0 * strides[0] +
                     i1 * strides[1] +
                     i2 * strides[2] +
                     i3 * strides[3];
        return data[offset];
    }

    // 只读版本
    const float &at_4d(int i0, int i1, int i2, int i3) const
    {
        int offset = i0 * strides[0] +
                     i1 * strides[1] +
                     i2 * strides[2] +
                     i3 * strides[3];
        return data[offset];
    }

    // ---------------------------------------------------------
    //  Contiguous: 内存连续化 (深拷贝 + 重排)
    // ---------------------------------------------------------
    Tensor contiguous() const
    {
        // 1. 创建一个形状一样的新张量
        // 注意：新张量的构造函数会自动计算标准的、连续的 strides
        Tensor out(this->shape);

        // 2. 目前只处理 4D 张量 (GPT 专用简化版)
        if (shape.size() == 4)
        {
            int B = shape[0];
            int NH = shape[1];
            int T = shape[2];
            int HS = shape[3];

            // 四重循环复制每一个元素
            for (int b = 0; b < B; ++b)
            {
                for (int nh = 0; nh < NH; ++nh)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        for (int hs = 0; hs < HS; ++hs)
                        {
                            // 这里有魔法：
                            // out.at_4d 得到新张量的标准 strides (顺序读写)
                            // this->at_4d 得到当前张量的 strides (可能乱序)
                            out.at_4d(b, nh, t, hs) = this->at_4d(b, nh, t, hs);
                        }
                    }
                }
            }
        }
        else
        {
            // 如果不是 4D 张量，直接拷贝 (偷懒，因为 GPT 只需要 4D 张量)
            // 严谨的话应该实现通用的递归复制
            out = *this;
            std::cerr << "Warning: contiguous only implemented for 4D tensors!" << std::endl;
        }

        return out;
    }

    // 掩码函数  用于注意力机制中的因果掩码

    void apply_causal_mask()
    {
        if (shape.size() != 4)
            return;

        int B = shape[0];
        int NH = shape[1];
        int T_row = shape[2];
        int T_col = shape[3]; // 通常 T_row == T_col

        float neg_inf = -std::numeric_limits<float>::infinity();

        for (int b = 0; b < B; ++b)
        {
            for (int nh = 0; nh < NH; ++nh)
            {
                for (int i = 0; i < T_row; ++i)
                {
                    for (int j = 0; j < T_col; ++j)
                    {
                        // 下三角逻辑：不能看到未来
                        if (j > i)
                        {
                            at_4d(b, nh, i, j) = neg_inf;
                        }
                    }
                }
            }
        }
    }

    // yolov5的激活函数
    Tensor Sigmoid() const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < result.data.size(); ++i)
        {
            float val = this->data[i];
            result.data[i] = 1.0f / (1.0f + std::exp(-val));
        }
        return result;
    }

    Tensor SiLU() const
    {
        Tensor result(this->shape);
        size_t size = this->data.size();
// 开启多线程
#pragma omp parallel for
        for (size_t i = 0; i < result.data.size(); ++i)
        {
            float val = this->data[i];
            float sigmoid = 1.0f / (1.0f + std::exp(-val));
            result.data[i] = val * sigmoid;
        }

        return result;
    }

    // 上采样
    Tensor Upsample() const
    {
        int N = shape[0];
        int C = shape[1];
        int H = shape[2];
        int W = shape[3];

        Tensor out({N, C, H * 2, W * 2});
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int i = 0; i < out.shape[2]; i++)
                {
                    for (int j = 0; j < out.shape[3]; j++)
                    {
                        float val = this->at_4d(n, c, i / 2, j / 2);
                        out.at_4d(n, c, i, j) = val;
                    }
                }
            }
        }

        return out;
    }

    // 拼接
    static Tensor Concat(const Tensor &t1, const Tensor &t2, int dim = 1)
    {
        if (t1.shape[2] != t2.shape[2] || t1.shape[3] != t2.shape[3])
        {
            std::cerr << "Concat Error: H and W must match!" << std::endl;
            exit(1);
        }

        int N = t1.shape[0];
        int C1 = t1.shape[1];
        int C2 = t2.shape[1];
        int H = t1.shape[2];
        int W = t1.shape[3];

        Tensor out({N, C1 + C2, H, W});

        int batch_size_A = C1 * H * W;
        int batch_size_B = C2 * H * W;
        int total_batch_size = (C1 + C2) * H * W;
        float *ptr = out.data.data();
        const float *ptr_A = t1.data.data();
        const float *ptr_B = t2.data.data();

        for (int n = 0; n < N; n++)
        {
            memcpy(ptr, ptr_A, batch_size_A * sizeof(float));

            ptr += batch_size_A;
            ptr_A += batch_size_A;

            memcpy(ptr, ptr_B, batch_size_B * sizeof(float));

            ptr += batch_size_B;
            ptr_B += batch_size_B;
        }

        return out;
    }

    Tensor Add(const Tensor &other) const
    {
        if (this->shape != other.shape)
        {
            std::cerr << "Add Error: Shape must match!" << std::endl;
            exit(1);
        }

        Tensor result(this->shape);
        // 获取原始指针
        const float *pA = this->data.data();
        const float *pB = other.data.data();
        float *pOut = result.data.data();
        size_t n = this->data.size();
        size_t i = 0;
        // AVX2指令集加速
        for (; i <= n - 8; i += 8)
        {
            // 搬运数据到256位寄存器
            __m256 va = _mm256_loadu_ps(pA + i);
            __m256 vb = _mm256_loadu_ps(pB + i);

            // 启用8路并行加法
            __m256 vsum = _mm256_add_ps(va, vb);

            // 回收内存
            _mm256_storeu_ps(pOut + i, vsum);
        }

        for (; i < n; i++)
        {
            pOut[i] = pA[i] + pB[i];
        }

        return result;
    }

    // 模拟 Int8 量化 (Float -> Int8 -> Float)
    // 目的：感受精度损失，预估上板效果
    Tensor FakeQuantizeInt8() const
    {
        Tensor result(this->shape);

        // 1. 找最大最小值 (寻找量化范围)
        float min_val = 1e9f;
        float max_val = -1e9f;
        for (float v : data)
        {
            if (v < min_val)
                min_val = v;
            if (v > max_val)
                max_val = v;
        }

        // 2. 计算缩放系数 (Scale) 和 零点 (Zero Point)
        // 我们要把 [min, max] 映射到 [-128, 127] (255个阶梯)
        float range = max_val - min_val;
        if (range == 0)
            range = 1.0f; // 防止除0

        float scale = range / 255.0f;
        float zero_point = -128.0f - (min_val / scale);

// 3. 逐元素量化再反量化
// 也可以加上 OpenMP 加速！
#pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i)
        {
            float real_val = data[i];

            // 量化: float -> int
            // round(x / S + Z)
            int q = std::round(real_val / scale + zero_point);

            // 截断 (Clamp) 到 [-128, 127]
            if (q < -128)
                q = -128;
            if (q > 127)
                q = 127;

            // 反量化: int -> float (这就不是原来的 float 了，是阶梯状的)
            // (q - Z) * S
            float deq_val = (q - zero_point) * scale;

            result.data[i] = deq_val;
        }

        return result;
    }

    void print()
    {
        std::cout << "Tensor shape={";
        for (size_t i = 0; i < shape.size(); ++i)
            std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
        std::cout << "}, strides={";
        for (size_t i = 0; i < strides.size(); ++i)
            std::cout << strides[i] << (i < strides.size() - 1 ? ", " : "");
        std::cout << "}\nData:\n";

        if (shape.size() == 2)
        {
            // 打印 2D 矩阵
            for (int i = 0; i < shape[0]; ++i)
            {
                std::cout << "[ ";
                for (int j = 0; j < shape[1]; ++j)
                {
                    int index = i * strides[0] + j * strides[1];
                    std::cout << std::fixed << std::setprecision(4) << data[index] << " ";
                }
                std::cout << "]\n";
            }
        }
        else
        {
            std::cout << "[ ";
            for (float v : data)
                std::cout << v << " ";
            std::cout << "]\n";
        }
    }
};

Tensor im2col(const Tensor &input, int k_h, int k_w, int stride = 1, int pad = 0)
{
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    // 计算输出特征图尺寸
    int out_h = (H + 2 * pad - k_h) / stride + 1;
    int out_w = (W + 2 * pad - k_w) / stride + 1;

    int rows = N * out_h * out_w;
    int cols = C * k_h * k_w;

    Tensor col({rows, cols});

    int row_index = 0;

    for (int n = 0; n < N; n++)
    {
        for (int i = 0; i < out_h; i++)
        {
            for (int j = 0; j < out_w; j++)
            {
                int col_index = 0;

                for (int c = 0; c < C; c++)
                {
                    for (int kh = 0; kh < k_h; kh++)
                    {
                        for (int kw = 0; kw < k_w; kw++)
                        {
                            int r = i * stride + kh - pad;
                            int col_pos = j * stride + kw - pad;

                            if (r >= 0 && r < H && col_pos >= 0 && col_pos < W)
                            {
                                col.at(row_index, col_index) = input.at_4d(n, c, r, col_pos);
                            }
                            else
                            {
                                col.at(row_index, col_index) = 0.0f;
                            }

                            col_index++;
                        }
                    }
                }
                row_index++;
            }
        }
    }

    return col;
}

// 卷积函数
Tensor conv2d(const Tensor &input, const Tensor &weight, const Tensor &bias, int stride = 1, int pad = 0)
{
    int N = input.shape[0];
    int FN = weight.shape[0]; // 滤波器的数量

    // weight.shape[1] 表示通道数 C
    int KH = weight.shape[2];
    int KW = weight.shape[3];

    Tensor col = im2col(input, KH, KW, stride, pad); // [N*out_h*out_w, C*KH*KW]
    col = col.Transpose(1, 0);                       // 转置为 [C*KH*KW, N*out_h*out_w] ,方便矩阵乘法

    int K_dim = weight.shape[1] * KH * KW;
    Tensor weight_flat = weight.view({FN, K_dim});

    Tensor out_flat = weight_flat.MatMul(col); // [FN, N*out_h*out_w]

    // 加上偏置
    int rows = out_flat.shape[0];
    int cols = out_flat.shape[1];

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            out_flat.at(i, j) += bias.data[i];
        }
    }

    // 计算输出特征图尺寸

    int out_h = (input.shape[2] + 2 * pad - KH) / stride + 1;
    int out_w = (input.shape[3] + 2 * pad - KW) / stride + 1;

    Tensor out = out_flat.view({FN, N, out_h, out_w});
    out = out.Transpose(1, 0);
    out = out.contiguous();

    return out;
}

// 池化函数 :找到卷积核覆盖区域中的最大值，同时记录最大值的位置，将原图尺寸缩小
Tensor max_pool2d(const Tensor &input, int pool_h, int pool_w, int stride = 2, int pad = 0)
{
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    // /// 1. 输出尺寸公式 (注意 2*pad)
    int out_h = (H + 2 * pad - pool_h) / stride + 1;
    int out_w = (W + 2 * pad - pool_w) / stride + 1;

    Tensor out({N, C, out_h, out_w});

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int i = 0; i < out_h; i++)
            {
                for (int j = 0; j < out_w; j++)
                {
                    float max_val = -FLT_MAX;

                    // 遍历池化窗口
                    for (int ki = 0; ki < pool_h; ki++)
                    {
                        for (int kj = 0; kj < pool_w; kj++)
                        {
                            // /// 2. 计算在原图上的坐标 (注意要减去 pad)
                            int r = i * stride + ki - pad;
                            int col_pos = j * stride + kj - pad;

                            float val;
                            // /// 3. 边界处理 (Hardcore Padding Logic)
                            // 如果坐标超出原图范围，则视为负无穷
                            if (r < 0 || r >= H || col_pos < 0 || col_pos >= W)
                            {
                                val = -FLT_MAX;
                            }
                            else
                            {
                                // 只有在范围内，才去取内存，否则 SegFault 警告
                                val = input.at_4d(n, c, r, col_pos);
                            }

                            if (val > max_val)
                            {
                                max_val = val;
                            }
                        }
                    }
                    out.at_4d(n, c, i, j) = max_val;
                }
            }
        }
    }
    return out;
}
