#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err__) << std::endl;     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int TILE = 16;

__global__ void gemmCoalesced32x8(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

__global__ void gemmTiledShared16x16(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    int num_tiles = (K + TILE - 1) / TILE;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int a_col = tile_idx * TILE + threadIdx.x;
        int b_row = tile_idx * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col]
            : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col]
            : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void cpuGemmRef(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int M,
                int N,
                int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

float maxAbsError(const std::vector<float>& x, const std::vector<float>& y) {
    float err = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        err = std::max(err, std::fabs(x[i] - y[i]));
    }
    return err;
}

template <typename Kernel>
float benchmarkKernel(Kernel kernel,
                      dim3 grid,
                      dim3 block,
                      const float* dA,
                      const float* dB,
                      float* dC,
                      int M,
                      int N,
                      int K,
                      int warmup,
                      int iters) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    int iters = 50;
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        iters = std::atoi(argv[4]);
    }

    std::cout << "[lesson3_gemm_tiled_shared] M=" << M
              << " N=" << N
              << " K=" << K
              << " iters=" << iters << "\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f), hRef(M * N, 0.0f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * hA.size()));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * hB.size()));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * hC.size()));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float) * hB.size(), cudaMemcpyHostToDevice));

    dim3 block_tiled(TILE, TILE);
    dim3 grid_tiled((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemmTiledShared16x16<<<grid_tiled, block_tiled>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost));

    cpuGemmRef(hA, hB, hRef, M, N, K);
    float err = maxAbsError(hC, hRef);
    std::cout << std::fixed << std::setprecision(6)
              << "[correctness] max_abs_err = " << err << "\n";

    dim3 block_coal(32, 8);
    dim3 grid_coal((N + block_coal.x - 1) / block_coal.x,
                   (M + block_coal.y - 1) / block_coal.y);
    float ms_coal = benchmarkKernel(gemmCoalesced32x8, grid_coal, block_coal,
                                    dA, dB, dC, M, N, K, 10, iters);

    float ms_tiled = benchmarkKernel(gemmTiledShared16x16, grid_tiled, block_tiled,
                                     dA, dB, dC, M, N, K, 10, iters);

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops_coal = (flops / (ms_coal / 1000.0)) / 1e9;
    double gflops_tiled = (flops / (ms_tiled / 1000.0)) / 1e9;

    std::cout << "[perf] coalesced(32x8):     " << ms_coal << " ms, "
              << gflops_coal << " GFLOPS\n";
    std::cout << "[perf] tiled_shared(16x16): " << ms_tiled << " ms, "
              << gflops_tiled << " GFLOPS\n";
    std::cout << "[perf] speedup: " << (ms_coal / ms_tiled) << "x vs Lesson2\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
