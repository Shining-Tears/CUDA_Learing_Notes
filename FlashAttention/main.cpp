#include <torch/torch.h>
#include <iostream>
#include <float.h>

#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) printf("%s: %lfms\n", #x, 1000 * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());

torch::Tensor multi_head_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

bool tensor_equal(torch::Tensor a, torch::Tensor b);

void flash_attention_v1(float* Q, float* K, float* V, float* L, float* M, float* O, int B, int nH, int N, int D, int Bc, int Br);

void flash_attention_v2(float* Q, float* K, float* V, float* O, int B, int nH, int N, int D, int Bc, int Br);

void init_data(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, torch::Tensor& l, torch::Tensor& m, torch::Tensor& o,
    const int batch_size, const int n_head, const int seq_len, const int head_embd) {
    l = torch::zeros({ batch_size, n_head, seq_len }).cuda();
    m = torch::full({ batch_size, n_head, seq_len }, -FLT_MAX).cuda();
    q = torch::rand({ batch_size, n_head, seq_len, head_embd }).cuda();
    k = torch::rand({ batch_size, n_head, seq_len, head_embd }).cuda();
    v = torch::rand({ batch_size, n_head, seq_len, head_embd }).cuda();
    o = torch::rand({ batch_size, n_head, seq_len, head_embd }).cuda();
}

int main() {
    const int batch_size = 16;
    const int n_head = 12;
    const int seq_len = 100;
    const int head_embd = 64;

    int qkv_space = batch_size * n_head * seq_len * head_embd * sizeof(float);
    int lm_space = batch_size * n_head * seq_len * sizeof(float);
    int Bc = 32, Br = 32;

    torch::Tensor q, k, v, l, m, o;

    init_data(q, k, v, l, m, o, batch_size, n_head, seq_len, head_embd);

    // d * N * N + d * N    N * N * d * d
    TICK(flash_attention_v1);
    flash_attention_v1(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), l.data_ptr<float>(), 
        m.data_ptr<float>(), o.data_ptr<float>(), batch_size, n_head, seq_len, head_embd, Bc, Br);
    TOCK(flash_attention_v1);

    TICK(flash_attention_v2);
    flash_attention_v2(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), o.data_ptr<float>(), 
        batch_size, n_head, seq_len, head_embd, Bc, Br);
    TOCK(flash_attention_v2);

    TICK(attention_base);
    torch::Tensor o_base = multi_head_attention(q, k, v);
    TOCK(attention_base);
    
    float tolerance = 1e-5;
    bool is_close = (torch::abs(o_base - o) < tolerance).all().item<bool>();
    if (is_close) {
        std::cout << "results close" << std::endl;
    }
    else {
        std::cout << "results not close" << std::endl;
    }

    system("pause");
    return 0;
}
