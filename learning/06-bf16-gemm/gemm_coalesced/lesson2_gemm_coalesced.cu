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

// Lesson1 baseline: 16x16
__global__ void gemmNaive16x16(const float* A, const float* B, float* C,
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

// Lesson2: warp-aligned mapping for better global-memory coalescing.
// blockDim.x = 32 makes each warp cover contiguous columns in one row.
__global__ void gemmCoalesced32x8(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;

        // Memory behavior:
        // - B[k, col] is contiguous across threadIdx.x in a warp (coalesced).
        // - A[row, k] is reused by threads in same row (cache/broadcast friendly).
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = acc;
    }
}

void cpuGemmRef(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int M, int N, int K) {
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
    float e = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        e = std::max(e, std::fabs(x[i] - y[i]));
    }
    return e;
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

    std::cout << "[lesson2_gemm_coalesced] M=" << M << " N=" << N << " K=" << K
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

    // correctness check with coalesced kernel
    dim3 block_coal(32, 8);
    dim3 grid_coal((N + block_coal.x - 1) / block_coal.x,
                   (M + block_coal.y - 1) / block_coal.y);
    gemmCoalesced32x8<<<grid_coal, block_coal>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost));

    cpuGemmRef(hA, hB, hRef, M, N, K);
    float err = maxAbsError(hC, hRef);
    std::cout << std::fixed << std::setprecision(6)
              << "[correctness] max_abs_err = " << err << "\n";

    // benchmark baseline (16x16)
    dim3 block_base(16, 16);
    dim3 grid_base((N + block_base.x - 1) / block_base.x,
                   (M + block_base.y - 1) / block_base.y);
    float ms_base = benchmarkKernel(gemmNaive16x16, grid_base, block_base,
                                    dA, dB, dC, M, N, K, 10, iters);

    // benchmark coalesced (32x8)
    float ms_coal = benchmarkKernel(gemmCoalesced32x8, grid_coal, block_coal,
                                    dA, dB, dC, M, N, K, 10, iters);

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops_base = (flops / (ms_base / 1000.0)) / 1e9;
    double gflops_coal = (flops / (ms_coal / 1000.0)) / 1e9;

    std::cout << "[perf] baseline(16x16):   " << ms_base << " ms, "
              << gflops_base << " GFLOPS\n";
    std::cout << "[perf] coalesced(32x8):  " << ms_coal << " ms, "
              << gflops_coal << " GFLOPS\n";
    std::cout << "[perf] speedup: " << (ms_base / ms_coal) << "x\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
