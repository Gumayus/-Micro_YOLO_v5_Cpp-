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

inline float random_normal()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dis(0.0f, 1.0f); // 标准正态分布，均值为0，方差和为1
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
        strides.resize(shape.size()); // 将步长向量调整为与shape向量相同的形状

        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride; // 设定最后一维的步长为1
            stride *= shape[i];  // 除去最后一维，其他维的步长等于当前维度的元素个数乘以前一维的步长
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

        data.resize(size, 0.0f); // 初始化张量每个元素为0

        compute_strides();
    }

    void fill(float value)
    {
        std::fill(data.begin(), data.end(), value); // 将data中的所有数都设置成value
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

        // 1. 形状完全一样 (Element-wise)
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
                float div = other.at(i, 0); // 取出这一行的除数
                for (int j = 0; j < cols; ++j)
                {
                    result.at(i, j) = this->at(i, j) / div;
                }
            }
            return result;
        }
        // 3. 【新增】4D 广播: [B, NH, T, D] / [B, NH, T, 1] (GPT Attention 刚需)
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
                        // 取出分母 (最后一维坐标是0)
                        float div = other.at_4d(b, nh, t, 0);
                        // 这一行的所有数都除以它
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
        int N = input_indices.size();    // 一次性查询的字符
        int embed_dim = weight.shape[1]; // 嵌入层的权重参数

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

        // 1. 基础检查
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

        // 2. 维度提取 (适用于 A[..., M, K] @ B[..., K, N])
        // 无论 2D 还是 4D，最后两维永远是矩阵乘法的核心
        int M = this->shape[ndim - 2];
        int K = this->shape[ndim - 1];
        int K_other = other.shape[ndim - 2];
        int N = other.shape[ndim - 1];

        // 3. K 维度必须对齐
        if (K != K_other)
        {
            std::cerr << "MatMul Shape Mismatch: "
                      << "A[..." << M << "," << K << "] @ B[..." << K_other << "," << N << "]" << std::endl;
            exit(1);
        }

        // 4. 检查 Batch 维度 (前 N-2 维必须一致)
        // 构造结果形状
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

        // 5. 创建结果张量
        Tensor out(out_shape);

        // =================================================
        //  路径 A: 2D 矩阵乘法 (Linear 层)
        // =================================================
        if (ndim == 2)
        {
// OpenMP 并行加速 (如果编译器支持)
#pragma omp parallel for collapse(2)
            for (int i = 0; i < M; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k)
                    {
                        // 使用 at(i, j) 自动处理 strides
                        sum += this->at(i, k) * other.at(k, j);
                    }
                    out.at(i, j) = sum;
                }
            }
        }
        // =================================================
        //  路径 B: 4D 批量矩阵乘法 (Attention 层)
        //  形状: [Batch, Head, Seq, Dim]
        // =================================================
        else if (ndim == 4)
        {
            int B_dim = shape[0];  // Batch Size
            int NH_dim = shape[1]; // Num Heads

// 四重循环太深？其实前两维是“并行”的，后三维是“计算”
#pragma omp parallel for collapse(2) // 并行处理每个 Batch 和 Head
            for (int b = 0; b < B_dim; ++b)
            {
                for (int nh = 0; nh < NH_dim; ++nh)
                {

                    // 这里是对每个 (b, nh) 下的矩阵做乘法
                    // [M, K] @ [K, N] -> [M, N]
                    for (int i = 0; i < M; ++i)
                    {
                        for (int j = 0; j < N; ++j)
                        {
                            float sum = 0.0f;
                            for (int k = 0; k < K; ++k)
                            {
                                // 核心魔法：使用 at_4d 穿透 stride 的迷雾
                                // 即使 K 被 transpose 转置过，at_4d 也能找到正确的位置
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
    // 张量捏合
    Tensor view(const std::vector<int> &new_shape) const // 重新捏合张量的shape
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

        // 1. 处理负数索引 (比如 -1 代表最后一维)
        if (axis < 0)
            axis += ndim;

        // 2. 目前只支持对“最后一维”求和 (Softmax/LayerNorm 刚需)
        if (axis != ndim - 1)
        {
            std::cerr << "Sum Error: Currently only supports summing over the LAST dimension!" << std::endl;
            exit(1);
        }

        // 3. 构造结果形状
        // 保持维度数不变，但最后一维变成 1 (KeepDim=True)
        std::vector<int> out_shape = this->shape;
        out_shape[axis] = 1;
        Tensor out(out_shape);

        // =================================================
        //  路径 A: 2D 情况 [rows, cols] -> [rows, 1]
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
        //  路径 B: 4D 情况 [B, NH, T, D] -> [B, NH, T, 1]
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
                        // 沿最后一维累加
                        for (int d = 0; d < D; ++d)
                        {
                            total += this->at_4d(b, nh, t, d);
                        }

                        // 存入结果 (最后一维坐标是 0)
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

    // 归一化

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

    // 层归一化  对最后一维进行操作
    Tensor LayerNorm(const Tensor &gamma, const Tensor &beta) const // gamma 缩放参数  beta 偏移参数
    {
        Tensor out(this->shape);

        int D = this->shape.back();
        int N = this->data.size() / D;

        const float eps = 1e-5f;

        for (int i = 0; i < N; ++i)
        {
            int offset = i * D; // 找到每一行的起始位置

            // 求均值 mean
            float sum = 0.0f;
            for (int j = 0; j < D; ++j)
            {
                sum += data[offset + j];
            }

            float mean = sum / D;

            // 求方差
            float sum_sq_diff = 0.0f;
            for (int j = 0; j < D; ++j)
            {
                float dff = data[offset + j] - mean;
                sum_sq_diff += dff * dff;
            }

            float variance = sum_sq_diff / D;

            // 准备标准化系数

            float inv_std = 1.0f / std::sqrt(variance + eps); // sqrt 计算平方根

            // 归一化

            for (int j = 0; j < D; ++j)
            {
                float n = (data[offset + j] - mean) * inv_std;

                out.data[offset + j] = n * gamma.data[j] + beta.data[j];
            }
        }

        return out;
    }

    Tensor Transpose(int dim0, int dim1) const // 传入需要交换的维度
    {
        Tensor out = *this;
        int ndim = out.shape.size();
        if (dim0 < 0)
            dim0 += ndim;

        if (dim1 < 0)
            dim1 += ndim;

        if (dim0 >= ndim || dim1 >= ndim) // 临界检查
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

    // 只读版
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
        // 注意：新张量的构造函数会自动计算出“标准的、连续的”strides
        Tensor out(this->shape);

        // 2. 只有 4D 张量才处理 (GPT 专用简化版)
        if (shape.size() == 4)
        {
            int B = shape[0];
            int NH = shape[1];
            int T = shape[2];
            int HS = shape[3];

            // 四重循环遍历每一个点
            for (int b = 0; b < B; ++b)
            {
                for (int nh = 0; nh < NH; ++nh)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        for (int hs = 0; hs < HS; ++hs)
                        {
                            // 【核心魔法】
                            // out.at_4d 用的是新张量的标准 strides (连续写)
                            // this->at_4d 用的是当前张量的乱 strides (跳着读)
                            out.at_4d(b, nh, t, hs) = this->at_4d(b, nh, t, hs);
                        }
                    }
                }
            }
        }
        else
        {
            // 如果不是 4D，暂时直接拷贝 (偷懒做法，反正 GPT 里主要是 4D 需要这个)
            // 严谨点应该报错或者写通用递归
            out = *this;
            std::cerr << "Warning: contiguous only implemented for 4D tensors!" << std::endl;
        }

        return out;
    }

    // 掩码函数  构造下三角矩阵用于聚合信息

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
                        // 核心逻辑：不能看未来
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
        for (size_t i = 0; i < result.data.size(); ++i)
        {
            float val = this->data[i];
            result.data[i] = 1.0f / (1.0f + std::exp(-val)) * val;
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
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = data[i] + other.data[i];
        }

        return result;
    }

    Tensor Sub(const Tensor &other) const
    {
        if (this->shape != other.shape)
        {
            std::cerr << "Sub Error: Shape must match!" << std::endl;
            exit(1);
        }

        Tensor result(this->shape);

        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = data[i] - other.data[i];
        }

        return result;
    }

    // 二维转置
    Tensor Transpose2D() const
    {
        if (this->shape.size() != 2)
        {
            std::cerr << "Transpose Error: Tensor must be 2D!" << std::endl;
            exit(1);
        }

        int H = this->shape[0];
        int W = this->shape[1];

        Tensor out({W, H});

        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < W; j++)
            {
                out.at(j, i) = this->at(i, j);
            }
        }

        return out;
    }

    // 二维矩阵求逆
    static Tensor Inverse2D(const Tensor &mat);

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

inline Tensor im2col(const Tensor &input, int k_h, int k_w, int stride = 1, int pad = 0)
{
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    // 计算输出图像尺寸
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

// 卷积操作
inline Tensor conv2d(const Tensor &input, const Tensor &weight, const Tensor &bias, int stride = 1, int pad = 0)
{
    int N = input.shape[0];
    int FN = weight.shape[0]; // 卷积核的数量

    // weight.shape[1] 表示通道数 C
    int KH = weight.shape[2];
    int KW = weight.shape[3];

    Tensor col = im2col(input, KH, KW, stride, pad); // [N*out_h*out_w, C*KH*KW]
    col = col.Transpose(1, 0);                       // 转置为 [C*KH*KW, N*out_h*out_w] ,配合矩阵乘法

    int K_idm = weight.shape[1] * KH * KW;
    Tensor weight_flat = weight.view({FN, K_idm});

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

    // 计算输出图像尺寸

    int out_h = (input.shape[2] + 2 * pad - KH) / stride + 1;
    int out_w = (input.shape[3] + 2 * pad - KW) / stride + 1;

    Tensor out = out_flat.view({FN, N, out_h, out_w});
    out = out.Transpose(1, 0);
    out = out.contiguous();

    return out;
}

// 池化操作 :找到卷积核覆盖区域的最大特征值，同时保留该特征值的位置，将原图像尺寸缩小
inline Tensor max_pool2d(const Tensor &input, int pool_h, int pool_w, int stride = 2, int pad = 0)
{
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    // /// 1. 修正输出尺寸公式 (加上 2*pad)
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

                    // 遍历池化核
                    for (int ki = 0; ki < pool_h; ki++)
                    {
                        for (int kj = 0; kj < pool_w; kj++)
                        {
                            // /// 2. 计算在原图上的坐标 (注意要减去 pad)
                            int r = i * stride + ki - pad;
                            int col_pos = j * stride + kj - pad;

                            float val;
                            // /// 3. 边界检查 (Hardcore Padding Logic)
                            // 如果坐标超出了原图范围，就认为它是负无穷
                            if (r < 0 || r >= H || col_pos < 0 || col_pos >= W)
                            {
                                val = -FLT_MAX;
                            }
                            else
                            {
                                // 只有在范围内，才敢去读内存，否则由 SegFault 伺候
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

inline Tensor Tensor::Inverse2D(const Tensor &mat)
{
    if (mat.shape[0] != 2 || mat.shape[1] != 2)
    {
        std::cerr << "Inverse2D Error: Shape size must 2x2!" << std::endl;
        exit(1);
    }

    float a = mat.data[0];
    float b = mat.data[1];
    float c = mat.data[2];
    float d = mat.data[3];

    float det = a * d - b * c;

    if (std::fabs(det) < 1e-6f)
    {
        std::cerr << "Inverse2D Error: Matrix is singular!" << std::endl;
        exit(1);
    }

    float inv_det = 1.0f / det;
    Tensor inv({2, 2});
    inv.data[0] = d * inv_det;
    inv.data[1] = -b * inv_det;
    inv.data[2] = -c * inv_det;
    inv.data[3] = a * inv_det;

    return inv;
}

class KalmanFilter
{
public:
    // dt : 时间间隔
    KalmanFilter(float dt = 0.033f);

    // 状态预测
    void predict();

    // 状态更新 观测值z [2x1]
    void update(const Tensor &z);

    // 获取当前状态估计值 [x,y,vx,vy]
    Tensor getState() const { return *X; }

private:
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
