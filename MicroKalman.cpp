#include "MicroKalman.h"
KalmanFilter::KalmanFilter(float dt)
{
    X = std::make_unique<Tensor>(std::vector<int>{4, 1});
    P = std::make_unique<Tensor>(std::vector<int>{4, 4});
    F = std::make_unique<Tensor>(std::vector<int>{4, 4});
    H = std::make_unique<Tensor>(std::vector<int>{2, 4});
    Q = std::make_unique<Tensor>(std::vector<int>{4, 4});
    R = std::make_unique<Tensor>(std::vector<int>{2, 2});
    I = std::make_unique<Tensor>(std::vector<int>{4, 4});

    init_matrices(dt);
}

void KalmanFilter::init_matrices(float dt)
{
    F->fill(0.0f);
    F->at(0, 0) = 1.0f;
    F->at(1, 1) = 1.0f;
    F->at(2, 2) = 1.0f;
    F->at(3, 3) = 1.0f;
    F->at(0, 2) = dt;
    F->at(1, 3) = dt;

    H->fill(0.0f);
    H->at(0, 0) = 1.0f;
    H->at(1, 1) = 1.0f;

    P->fill(0.0f);
    P->at(0, 0) = 1.0f;
    P->at(1, 1) = 1.0f;
    P->at(2, 2) = 1.0f;
    P->at(3, 3) = 1.0f;

    I->fill(0.0f);
    I->at(0, 0) = 1.0f;
    I->at(1, 1) = 1.0f;
    I->at(2, 2) = 1.0f;
    I->at(3, 3) = 1.0f;

    Q->fill(0.0f);
    float q = 0.01f; // 过程噪声强度
    Q->at(0, 0) = q;
    Q->at(1, 1) = q;
    Q->at(2, 2) = q;
    Q->at(3, 3) = q;

    float r = 0.1f; // 测量噪声强度
    R->fill(0.0f);
    R->at(0, 0) = r;
    R->at(1, 1) = r;
}

void KalmanFilter::predict()
{
    Tensor x_pred = F->MatMul(*X);
    *X = x_pred; // 状态更新
    Tensor Ft = F->Transpose2D();
    Tensor FPFt = F->MatMul(*P).MatMul(Ft);
    *P = FPFt.Add(*Q); // 协方差更新
}

void KalmanFilter::update(const Tensor &z)
{
    Tensor H_T = H->Transpose2D();
    // 卡尔曼增益计算
    Tensor P_HT = P->MatMul(H_T);
    Tensor HP_HT = H->MatMul(P_HT);
    Tensor S = HP_HT.Add(*R);

    Tensor S_inv = Tensor::Inverse2D(S);
    Tensor K = P_HT.MatMul(S_inv);

    // 更新状态X
    Tensor Hx = H->MatMul(*X);
    Tensor y = z.Sub(Hx); // 测量残差
    Tensor Ky = K.MatMul(y);
    *X = X->Add(Ky);

    // 更新协方差P
    Tensor KH = K.MatMul(*H);
    Tensor I_KH = I->Sub(KH);
    *P = I_KH.MatMul(*P);
}

void KalmanFilter::set_state(float init_x, float init_y, float init_vx, float init_vy)
{
    X->data[0] = init_x;
    X->data[1] = init_y;
    X->data[2] = init_vx;
    X->data[3] = init_vy;

    // 重置协方差矩阵P
    P->fill(0.0f);
    P->at(0, 0) = 1.0f;
    P->at(1, 1) = 1.0f;
    P->at(2, 2) = 1.0f;
    P->at(3, 3) = 1.0f;
}
