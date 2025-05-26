#include <torch/torch.h>
#include <iostream>
#include <float.h>

#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) printf("%s: %lfms\n", #x, 1000 * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());

void cublas_sgemm(float *a, float *b, float *c, size_t M, size_t N, size_t K);

bool tensor_equal(torch::Tensor a, torch::Tensor b);

void my_sgemm(float *a, float *b, float *c, const int M, const int N, const int K);

void init_data(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c1, torch::Tensor& c2, 
    const int M, const int N, const int K) {
    a = torch::rand({ M, K }).cuda();
    b = torch::rand({ K, N }).cuda();
    c1 = torch::rand({ M, N }).cuda();
    c2 = torch::rand({ M, N }).cuda();
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 2048;
    torch::Tensor a, b, c1, c2;

    init_data(a, b, c1, c2, M, N, K);

    // 预热
    for (int i = 0;  i < 100; i++) {
        cublas_sgemm(a.data_ptr<float>(), b.data_ptr<float>(), c1.data_ptr<float>(), M, N, K);
    }
    TICK(cublas_sgemm);
    for (int i = 0;  i < 100; i++) {
        cublas_sgemm(a.data_ptr<float>(), b.data_ptr<float>(), c1.data_ptr<float>(), M, N, K);
    }
    TOCK(cublas_sgemm);

    // 预热
    for (int i = 0;  i < 100; i++) {
        my_sgemm(a.data_ptr<float>(), b.data_ptr<float>(), c2.data_ptr<float>(), M, N, K);
    }
    TICK(my_sgemm);
    for (int i = 0;  i < 100; i++) {
        my_sgemm(a.data_ptr<float>(), b.data_ptr<float>(), c2.data_ptr<float>(), M, N, K);
    }
    TOCK(my_sgemm);

    float tolerance = 1e-5;
    bool is_close = (torch::abs(c1 - c2) < tolerance).all().item<bool>();
    if (is_close) {
        std::cout << "results close" << std::endl;
    }
    else {
        std::cout << "results not close" << std::endl;
    }

    system("pause");
    return 0;
}
