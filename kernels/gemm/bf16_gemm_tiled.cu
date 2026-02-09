#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int TILE = 16;

__global__ void bf16GemmTiledKernel(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                    float* C, int M, int N, int K) {
    __shared__ __nv_bfloat16 As[TILE][TILE];
    __shared__ __nv_bfloat16 Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : __float2bfloat16(0.0f);
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += __bfloat162float(As[threadIdx.y][k]) * __bfloat162float(Bs[k][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

int main(int argc, char** argv) {
    int M = 256, N = 256, K = 256;
    if (argc == 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }

    std::vector<__nv_bfloat16> h_A(M * K), h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16(((i % 17) - 8) * 0.05f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16(((i % 11) - 5) * 0.05f);

    __nv_bfloat16 *d_A = nullptr, *d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(__nv_bfloat16) * M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(__nv_bfloat16) * K * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * M * N));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(__nv_bfloat16) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(__nv_bfloat16) * K * N, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    bf16GemmTiledKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    std::cout << "[bf16_gemm_tiled] done. shape=" << M << "x" << N << "x" << K << "\n";
    std::cout << "C[0], C[mid], C[last] = "
              << h_C[0] << ", " << h_C[(M * N) / 2] << ", " << h_C[M * N - 1] << "\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
