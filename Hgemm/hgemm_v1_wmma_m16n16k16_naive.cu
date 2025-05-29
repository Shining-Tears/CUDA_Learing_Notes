#include <mma.h>

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(half* A, half* B, half* C, 
    int M, int N, int K) {
    const int NUM_K_TILES = (K + WMMA_K - 1) / K;
    const int load_gmem_a_m = blockIdx.y * WMMA_M;
    const int load_gmem_b_n = blockIdx.x * WMMA_N;
    if (load_gmem_a_m >= M && load_gmem_b_n >= N) {
        return;
    }
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    nvcuda::wmma::fill_fragment(C_frag, 0.0);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> A_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> B_frag;

    for (int k = 0; k < NUM_K_YILES; ++k) {
        nvcuda::wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
        nvcuda::wmma::load_matrix_sync(B_frag, B + (k * WMMA_K) * N + load_gmem_b_m, N);

        nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N, nvcuda::wmma::mem_row_major);

}