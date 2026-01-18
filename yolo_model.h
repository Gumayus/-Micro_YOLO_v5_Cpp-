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

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

class YoloV5Detector {
public:
    
    YoloV5Detector(const std::string& modelPath, bool isCuda = false);
    
    std::vector<Detection> detect(cv::Mat& frame,bool use_quant = false);

private:
    // ����
    const float INPUT_W = 640.0;
    const float INPUT_H = 640.0;
    const float SCORE_THRES = 0.15f; // 置信度阈值
    const float NMS_THRES = 0.45f;   // NMS 阈值

    cv::dnn::Net net;

    
    cv::Mat formatToSquare(const cv::Mat& source);
};


inline float random_normal()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dis(0.0f, 1.0f); 
    return dis(gen);
}

class Tensor
{
public:
    std::vector<float> data;
    std::vector<int> shape;   
    std::vector<int> strides; 

    void compute_strides()
    {
        strides.resize(shape.size()); 
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride; 
            stride *= shape[i]; 
        }
    }

    Tensor() {}

    static Tensor randn(const std::vector<int>& shape)
    {
        Tensor t{ shape };
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

        data.resize(size, 0.0f); 

        compute_strides();
    }

    void fill(float value)
    {
        std::fill(data.begin(), data.end(), value);
    }

    Tensor operator+(const Tensor& other) const
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

    Tensor operator-(const Tensor& other) const
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

    Tensor operator/(const Tensor& other) const
    {
        int ndim = this->shape.size();

        // 1. ��״��ȫһ�� (Element-wise)
        if (this->shape == other.shape) {
            Tensor result(this->shape);
            for (size_t i = 0; i < data.size(); ++i) {
                result.data[i] = data[i] / other.data[i];
            }
            return result;
        }
        // 2. 2D �㲥: [M, N] / [M, 1]
        else if (ndim == 2 && other.shape.size() == 2 &&
            other.shape[1] == 1 && this->shape[0] == other.shape[0])
        {
            Tensor result(this->shape);
            int rows = shape[0]; int cols = shape[1];
            for (int i = 0; i < rows; ++i) {
                float div = other.at(i, 0); // ȡ����һ�еĳ���
                for (int j = 0; j < cols; ++j) {
                    result.at(i, j) = this->at(i, j) / div;
                }
            }
            return result;
        }
        // 3. ��������4D �㲥: [B, NH, T, D] / [B, NH, T, 1] (GPT Attention ����)
        else if (ndim == 4 && other.shape.size() == 4 &&
            other.shape[3] == 1 && // ���һά�� 1
            this->shape[0] == other.shape[0] &&
            this->shape[1] == other.shape[1] &&
            this->shape[2] == other.shape[2])
        {
            Tensor result(this->shape);
            int B = shape[0]; int NH = shape[1]; int T = shape[2]; int D = shape[3];

            for (int b = 0; b < B; ++b) {
                for (int nh = 0; nh < NH; ++nh) {
                    for (int t = 0; t < T; ++t) {
                        // ȡ����ĸ (���һά������0)
                        float div = other.at_4d(b, nh, t, 0);
                        // ��һ�е���������������
                        for (int d = 0; d < D; ++d) {
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

    float& at(int i, int j)
    {
        return data[i * strides[0] + j * strides[1]];
    }

    const float& at(int i, int j) const
    {
        return data[i * strides[0] + j * strides[1]];
    }

    // Ƕ���
    static Tensor Embedding(const std::vector<int>& input_indices, const Tensor& weight)
    {
        int N = input_indices.size();    // һ���Բ�ѯ���ַ�
        int embed_dim = weight.shape[1]; // Ƕ����Ȩ�ز���

        Tensor out({ N, embed_dim });

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

    // ����˷�
    Tensor MatMul(const Tensor& other) const
    {
        int ndim = this->shape.size();

        // 1. �������
        if (ndim != other.shape.size()) {
            std::cerr << "MatMul Error: Ranks must match! " << ndim << " vs " << other.shape.size() << std::endl;
            exit(1);
        }
        if (ndim != 2 && ndim != 4) {
            std::cerr << "MatMul Error: Currently only supports 2D or 4D tensors (for GPT)." << std::endl;
            exit(1);
        }

        // 2. ά����ȡ (������ A[..., M, K] @ B[..., K, N])
        // ���� 2D ���� 4D�������ά��Զ�Ǿ���˷��ĺ���
        int M = this->shape[ndim - 2];
        int K = this->shape[ndim - 1];
        int K_other = other.shape[ndim - 2];
        int N = other.shape[ndim - 1];

        // 3. K ά�ȱ������
        if (K != K_other) {
            std::cerr << "MatMul Shape Mismatch: "
                << "A[..." << M << "," << K << "] @ B[..." << K_other << "," << N << "]" << std::endl;
            exit(1);
        }

        // 4. ��� Batch ά�� (ǰ N-2 ά����һ��)
        // ��������״
        std::vector<int> out_shape = this->shape;
        out_shape[ndim - 1] = N; // ���һά��� N

        for (int i = 0; i < ndim - 2; ++i) {
            if (this->shape[i] != other.shape[i]) {
                std::cerr << "MatMul Batch Dimension Mismatch!" << std::endl;
                exit(1);
            }
        }

        // 5. �����������
        Tensor out(out_shape);

        // =================================================
        //  ·�� A: 2D ����˷� (Linear ��)
        // =================================================
        if (ndim == 2)
        {
            // OpenMP ���м��� (���������֧��)
#pragma omp parallel for collapse(2)
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        // ʹ�� at(i, j) �Զ����� strides
                        sum += this->at(i, k) * other.at(k, j);
                    }
                    out.at(i, j) = sum;
                }
            }
        }
        // =================================================
        //  ·�� B: 4D ��������˷� (Attention ��)
        //  ��״: [Batch, Head, Seq, Dim]
        // =================================================
        else if (ndim == 4)
        {
            int B_dim = shape[0];  // Batch Size
            int NH_dim = shape[1]; // Num Heads

            // ����ѭ��̫���ʵǰ��ά�ǡ����С��ģ�����ά�ǡ����㡱
#pragma omp parallel for collapse(2) // ���д���ÿ�� Batch �� Head
            for (int b = 0; b < B_dim; ++b) {
                for (int nh = 0; nh < NH_dim; ++nh) {

                    // �����Ƕ�ÿ�� (b, nh) �µľ������˷�
                    // [M, K] @ [K, N] -> [M, N]
                    for (int i = 0; i < M; ++i) {
                        for (int j = 0; j < N; ++j) {
                            float sum = 0.0f;
                            for (int k = 0; k < K; ++k) {
                                // ����ħ����ʹ�� at_4d ��͸ stride ������
                                // ��ʹ K �� transpose ת�ù���at_4d Ҳ���ҵ���ȷ��λ��
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
    // �������
    Tensor view(const std::vector<int>& new_shape) const // �������������shape
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

    // �����

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

        // 1. ������������ (���� -1 �������һά)
        if (axis < 0) axis += ndim;

        // 2. Ŀǰֻ֧�ֶԡ����һά����� (Softmax/LayerNorm ����)
        if (axis != ndim - 1) {
            std::cerr << "Sum Error: Currently only supports summing over the LAST dimension!" << std::endl;
            exit(1);
        }

        // 3. ��������״
        // ����ά�������䣬�����һά��� 1 (KeepDim=True)
        std::vector<int> out_shape = this->shape;
        out_shape[axis] = 1;
        Tensor out(out_shape);

        // =================================================
        //  ·�� A: 2D ��� [rows, cols] -> [rows, 1]
        // =================================================
        if (ndim == 2)
        {
            int rows = shape[0];
            int cols = shape[1];

            for (int i = 0; i < rows; ++i) {
                float total = 0.0f;
                for (int j = 0; j < cols; ++j) {
                    total += this->at(i, j);
                }
                out.at(i, 0) = total;
            }
        }
        // =================================================
        //  ·�� B: 4D ��� [B, NH, T, D] -> [B, NH, T, 1]
        // =================================================
        else if (ndim == 4)
        {
            int B = shape[0];
            int NH = shape[1];
            int T = shape[2];
            int D = shape[3]; // ���һά

            // ����ǰ��ά
            for (int b = 0; b < B; ++b) {
                for (int nh = 0; nh < NH; ++nh) {
                    for (int t = 0; t < T; ++t) {

                        float total = 0.0f;
                        // �����һά�ۼ�
                        for (int d = 0; d < D; ++d) {
                            total += this->at_4d(b, nh, t, d);
                        }

                        // ������ (���һά������ 0)
                        out.at_4d(b, nh, t, 0) = total;
                    }
                }
            }
        }
        else {
            std::cerr << "Sum Error: Only 2D and 4D tensors supported." << std::endl;
            exit(1);
        }

        return out;
    }

    //��һ��

    Tensor softmax(int axis = -1) const
    {
        Tensor exps = this->exp();

        Tensor sum = exps.sum(axis);

        return exps / sum;
    }

    // GPT�ļ����
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

    // ���һ��  �����һά���в���
    Tensor LayerNorm(const Tensor& gamma, const Tensor& beta) const // gamma ���Ų���  beta ƫ�Ʋ���
    {
        Tensor out(this->shape);

        int D = this->shape.back();
        int N = this->data.size() / D;

        const float eps = 1e-5f;

        for (int i = 0; i < N; ++i)
        {
            int offset = i * D; // �ҵ�ÿһ�е���ʼλ��

            // ���ֵ mean
            float sum = 0.0f;
            for (int j = 0; j < D; ++j)
            {
                sum += data[offset + j];
            }

            float mean = sum / D;

            // �󷽲�
            float sum_sq_diff = 0.0f;
            for (int j = 0; j < D; ++j)
            {
                float dff = data[offset + j] - mean;
                sum_sq_diff += dff * dff;
            }

            float variance = sum_sq_diff / D;

            // ׼����׼��ϵ��

            float inv_std = 1.0f / std::sqrt(variance + eps); // sqrt ����ƽ����

            // ��һ��

            for (int j = 0; j < D; ++j)
            {
                float n = (data[offset + j] - mean) * inv_std;

                out.data[offset + j] = n * gamma.data[j] + beta.data[j];
            }
        }

        return out;
    }

    Tensor Transpose(int dim0, int dim1) const // ������Ҫ������ά��
    {
        Tensor out = *this;
        int ndim = out.shape.size();
        if (dim0 < 0)
            dim0 += ndim;

        if (dim1 < 0)
            dim1 += ndim;

        if (dim0 >= ndim || dim1 >= ndim)  //�ٽ���
        {
            std::cerr << "Error: Transpose dim out of bounds!" << std::endl;
            exit(1);
        }

        std::swap(out.shape[dim0], out.shape[dim1]); // ������״

        std::swap(out.strides[dim0], out.strides[dim1]); // ��������

        return out;
    }

    float& at_4d(int i0, int i1, int i2, int i3) {
        int offset = i0 * strides[0] +
            i1 * strides[1] +
            i2 * strides[2] +
            i3 * strides[3];
        return data[offset];
    }

    // ֻ����
    const float& at_4d(int i0, int i1, int i2, int i3) const {
        int offset = i0 * strides[0] +
            i1 * strides[1] +
            i2 * strides[2] +
            i3 * strides[3];
        return data[offset];
    }

    // ---------------------------------------------------------
    //  Contiguous: �ڴ������� (��� + ����)
    // ---------------------------------------------------------
    Tensor contiguous() const {
        // 1. ����һ����״һ����������
        // ע�⣺�������Ĺ��캯�����Զ����������׼�ġ������ġ�strides
        Tensor out(this->shape);

        // 2. ֻ�� 4D �����Ŵ��� (GPT ר�ü򻯰�)
        if (shape.size() == 4) {
            int B = shape[0];
            int NH = shape[1];
            int T = shape[2];
            int HS = shape[3];

            // ����ѭ������ÿһ����
            for (int b = 0; b < B; ++b) {
                for (int nh = 0; nh < NH; ++nh) {
                    for (int t = 0; t < T; ++t) {
                        for (int hs = 0; hs < HS; ++hs) {
                            // ������ħ����
                            // out.at_4d �õ����������ı�׼ strides (����д)
                            // this->at_4d �õ��ǵ�ǰ�������� strides (���Ŷ�)
                            out.at_4d(b, nh, t, hs) = this->at_4d(b, nh, t, hs);
                        }
                    }
                }
            }
        }
        else {
            // ������� 4D����ʱֱ�ӿ��� (͵������������ GPT ����Ҫ�� 4D ��Ҫ���)
            // �Ͻ���Ӧ�ñ�������дͨ�õݹ�
            out = *this;
            std::cerr << "Warning: contiguous only implemented for 4D tensors!" << std::endl;
        }

        return out;
    }

    //���뺯��  ���������Ǿ������ھۺ���Ϣ

    void apply_causal_mask()
    {
        if (shape.size() != 4) return;

        int B = shape[0]; int NH = shape[1];
        int T_row = shape[2]; int T_col = shape[3]; // ͨ�� T_row == T_col

        float neg_inf = -std::numeric_limits<float>::infinity();

        for (int b = 0; b < B; ++b) {
            for (int nh = 0; nh < NH; ++nh) {
                for (int i = 0; i < T_row; ++i) {
                    for (int j = 0; j < T_col; ++j) {
                        // �����߼������ܿ�δ��
                        if (j > i) {
                            at_4d(b, nh, i, j) = neg_inf;
                        }
                    }
                }
            }
        }
    }

    //yolov5�ļ����
    Tensor Sigmoid()const
    {
        Tensor result(this->shape);
        for (size_t i = 0; i < result.data.size(); ++i)
        {
            float val = this->data[i];
            result.data[i] = 1.0f / (1.0f + std::exp(-val));
        }
        return result;
    }

    Tensor SiLU()const
    {
        Tensor result(this->shape);
        size_t size = this->data.size();
        //开启多线程
        # pragma omp parallel for
        for (size_t i = 0; i < result.data.size(); ++i)
        {
            float val = this->data[i];
            float sigmoid = 1.0f / (1.0f + std::exp(-val));
            result.data[i] = val * sigmoid;
        }

        return result;
    }

    //�ϲ���
    Tensor Upsample() const
    {
        int N = shape[0];
        int C = shape[1];
        int H = shape[2];
        int W = shape[3];

        Tensor out({ N,C,H * 2,W * 2 });
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

    //ƴ��
    static Tensor  Concat(const Tensor& t1, const Tensor& t2, int dim = 1)
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

        Tensor out({ N,C1 + C2,H,W });

        int batch_size_A = C1 * H * W;
        int batch_size_B = C2 * H * W;
        int total_batch_size = (C1 + C2) * H * W;
        float* ptr = out.data.data();
        const float* ptr_A = t1.data.data();
        const float* ptr_B = t2.data.data();

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

    Tensor Add(const Tensor& other) const
    {
        if (this->shape != other.shape)
        {
            std::cerr << "Add Error: Shape must match!" << std::endl;
            exit(1);
        }

        Tensor result(this->shape);
        //获取原始指针
        const float *pA = this->data.data();
        const float *pB = other.data.data();
        float *pOut = result.data.data();
        size_t n =this->data.size();
        size_t i =0;
        //AVX2指令集加速
        for (;i<=n-8;i+=8)
        {
            //搬运数据到256位寄存器
            __m256 va = _mm256_loadu_ps(pA + i);
            __m256 vb = _mm256_loadu_ps(pB + i);

            //启用8路并行加法
            __m256 vsum = _mm256_add_ps(va, vb);

            //回收内存
             _mm256_storeu_ps(pOut + i, vsum);
        }

        for(;i<n;i++)
        {
            pOut[i] = pA[i] + pB[i];
        }

        return result;
    }

    // 模拟 Int8 量化 (Float -> Int8 -> Float)
    // 目的：感受精度损失，预估上板效果
    Tensor FakeQuantizeInt8() const {
        Tensor result(this->shape);
        
        // 1. 找最大最小值 (寻找量化范围)
        float min_val = 1e9f;
        float max_val = -1e9f;
        for(float v : data) {
            if(v < min_val) min_val = v;
            if(v > max_val) max_val = v;
        }

        // 2. 计算缩放系数 (Scale) 和 零点 (Zero Point)
        // 我们要把 [min, max] 映射到 [-128, 127] (255个阶梯)
        float range = max_val - min_val;
        if(range == 0) range = 1.0f; // 防止除0
        
        float scale = range / 255.0f;
        float zero_point = -128.0f - (min_val / scale);

        // 3. 逐元素量化再反量化
        // 也可以加上 OpenMP 加速！
        #pragma omp parallel for
        for(size_t i=0; i<data.size(); ++i) {
            float real_val = data[i];

            // 量化: float -> int
            // round(x / S + Z)
            int q = std::round(real_val / scale + zero_point);
            
            // 截断 (Clamp) 到 [-128, 127]
            if (q < -128) q = -128;
            if (q > 127)  q = 127;

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
            // ��ӡ 2D ����
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

Tensor im2col(const Tensor& input, int k_h, int k_w, int stride = 1, int pad = 0)
{
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    //�������ͼ��ߴ�
    int out_h = (H + 2 * pad - k_h) / stride + 1;
    int out_w = (W + 2 * pad - k_w) / stride + 1;

    int rows = N * out_h * out_w;
    int cols = C * k_h * k_w;

    Tensor col({ rows,cols });

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

//��������
Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias, int stride = 1, int pad = 0)
{
    int N = input.shape[0];
    int FN = weight.shape[0]; // �����˵�����

    //weight.shape[1] ��ʾͨ���� C
    int KH = weight.shape[2];
    int KW = weight.shape[3];

    Tensor col = im2col(input, KH, KW, stride, pad); // [N*out_h*out_w, C*KH*KW]
    col = col.Transpose(1, 0); //ת��Ϊ [C*KH*KW, N*out_h*out_w] ,��Ͼ���˷�

    int K_idm = weight.shape[1] * KH * KW;
    Tensor weight_flat = weight.view({ FN,K_idm });

    Tensor out_flat = weight_flat.MatMul(col); // [FN, N*out_h*out_w]

    //����ƫ��
    int rows = out_flat.shape[0];
    int cols = out_flat.shape[1];

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            out_flat.at(i, j) += bias.data[i];
        }
    }

    //�������ͼ��ߴ�

    int out_h = (input.shape[2] + 2 * pad - KH) / stride + 1;
    int out_w = (input.shape[3] + 2 * pad - KW) / stride + 1;

    Tensor out = out_flat.view({ FN,N,out_h,out_w });
    out = out.Transpose(1, 0);
    out = out.contiguous();

    return out;
}

//�ػ����� :�ҵ������˸���������������ֵ��ͬʱ����������ֵ��λ�ã���ԭͼ��ߴ���С
Tensor max_pool2d(const Tensor& input, int pool_h, int pool_w, int stride = 2, int pad = 0)
{
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    // /// 1. ��������ߴ繫ʽ (���� 2*pad)
    int out_h = (H + 2 * pad - pool_h) / stride + 1;
    int out_w = (W + 2 * pad - pool_w) / stride + 1;

    Tensor out({ N, C, out_h, out_w });

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int i = 0; i < out_h; i++)
            {
                for (int j = 0; j < out_w; j++)
                {
                    float max_val = -FLT_MAX;

                    // �����ػ���
                    for (int ki = 0; ki < pool_h; ki++)
                    {
                        for (int kj = 0; kj < pool_w; kj++)
                        {
                            // /// 2. ������ԭͼ�ϵ����� (ע��Ҫ��ȥ pad)
                            int r = i * stride + ki - pad;
                            int col_pos = j * stride + kj - pad;

                            float val;
                            // /// 3. �߽��� (Hardcore Padding Logic)
                            // ������곬����ԭͼ��Χ������Ϊ���Ǹ�����
                            if (r < 0 || r >= H || col_pos < 0 || col_pos >= W) {
                                val = -FLT_MAX;
                            }
                            else {
                                // ֻ���ڷ�Χ�ڣ��Ÿ�ȥ���ڴ棬������ SegFault �ź�
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

