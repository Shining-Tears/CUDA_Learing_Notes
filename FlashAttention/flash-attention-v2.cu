#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void flash_attention_kernel_v2(const float* Q, const float* K, const float* V, const int N, const int D, \
									 const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale, \
									 float* O) {

	int tx = threadIdx.x;                 // thread id
	int bx = blockIdx.x, by = blockIdx.y; // batch id, head id

	extern __shared__ float sram[];
	const int qkv_offset = (bx * gridDim.y * N * D) + (by * N * D);   // q k v matrix offset
	const int q_tile_size = Br * D;       // Q Matrix tile size
	const int kv_tile_size = Bc * D;      // K,V Matrix tile size
	float* Qi = sram;                     // shared memory cache Q Matrix
	float* Kj = sram + q_tile_size;       // shared memory cache K Matrix
	float* Vj = Kj + kv_tile_size;        // shared memory cache V Matrix
	float* Ss = Vj + kv_tile_size;        // shared memory cache Q * K^T Matrix

	// block start q/k/v/o/l/m
	Q = Q + qkv_offset;
	K = K + qkv_offset;
	V = V + qkv_offset;
	O = O + qkv_offset;

	// loop Q Matrix
	for (int i = 0; i < Tr; j++) {
        const int Br_new = min(M, (i + 1) * Br) - i * Br;
		float row_m_prev = -FLT_MAX;
        float row_l_prev = 0;
		// Cache Q Matrix
		const float* Q_start = Q + q_tile_size * i;
		for (int row = tx; row < Br_new; row += blockDim.x) {
			for (int d = 0; d < D; d++) {
				Qi[row * D + d] = Q_start[row * D + d];
			}
		}
		__syncthreads();
		for (int j = 0; j < Tc; j++) {
			const int Bc_new = min(N, (j + 1) * Bc) - j * Bc;
			const float* K_start = K + kv_tile_size * j;
			const float* V_start = V + kv_tile_size * j;
			// cache K V Matrix
			for (int row = tx; row < Bc_new; row += blockDim.x) {
				for (int d = 0; d < D; d++) {
					Kj[row * D + d] = K_start[row * D + d];
					Vj[row * D + d] = V_start[row * D + d];
				}
			}
			__syncthreads();

			if (Br * i + tx < M) {
				float row_m = -FLT_MAX;
                float row_l = 0.f;
				for (int y = 0; y < Bc_new; y++) {
                    float sum = 0.f;
                    for (int d = 0; d < D; d++) {
                        sum += Qi[tx * D + d] * Kj[y * D + d];
                    }
                    sum *= softmax_scale;
                    Ss[(tx * Bc_new) + y] = sum;
                    row_m = max(row_m, sum);
                }
                for (int y = 0; y < Bc_new; y++) {
                    Ss[(tx * Bc_new) + y] = __expf(Ss[(tx * Bc_new) + y] - row_m);
                    row_l += Ss[(tx * Bc_new) + y];
                }
                float row_m_new = max(row_m, row_m_prev);
                float row_l_new = row_l_prev * __expf(row_m_prev - row_m_new) + row_l * __expf(row_m - row_m_new);
    
                for (int d = 0; d < D; d++) {
                    float pv = 0.f;
                    for (int y = 0; y < Bc_new; y++) {
                        pv += Ss[(tx * Bc_new) + y] * Vj[((D * y) + d)]; // e^(xn - row_m) * V
                    }
    
                    O_start[(tx * D) + d] = (1 / row_l_new) *
                        (O_start[(tx * D) + d] * row_l_prev * __expf(row_m_prev - row_m_new) 
                        + pv * __expf(row_m - row_m_new));
                }
				row_m_prev = row_m_new;
                row_l_prev = row_l_new;
			}
			__syncthreads();
		}
	}
}

torch::Tensor multi_head_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    torch::Tensor att = torch::matmul(Q, K.transpose(-2, -1));
    att = att * (1.0 / std::sqrt(K.size(-1)));

    att = torch::nn::functional::softmax(att, -1);
    att = torch::matmul(att, V);

	return att;
}

bool tensor_equal(torch::Tensor a, torch::Tensor b) {
    return ((torch::abs(a - b)) < 1e-4).all().item<bool>();
}

void flash_attention_v2(float* Q, float* K, float* V, float* O, int B, int nH, int N, 
    int D, int Bc, int Br) {
	const int Tc = ceil((float)N / Bc); const int Tr = ceil((float)N / Br);
	const float softmax_scale = 1.0 / std::sqrt(D);

	dim3 grid_dim(B, nH);
	dim3 block_dim(Br);

	const int sram_size = (2 * D * Bc * sizeof(float)) + (D * Br * sizeof(float)) + (Bc * Br * sizeof(float));
	int max_sram_size;
	cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
	printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

	flash_attention_kernel_v2 << <grid_dim, block_dim, sram_size >> > (
		Q, K, V,
		N, D, Tc, Tr, Bc, Br, softmax_scale,
		O);

	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();

	std::cout << cudaGetErrorString(error) << std::endl;
	return;
}