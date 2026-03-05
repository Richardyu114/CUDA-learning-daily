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

// 每个线程负责计算一个 C[row, col]
__global__ void naiveGemmKernel(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        // 逐 K 维做点积
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

int main(int argc, char** argv) {
    // 默认配置：适合 lesson1 本地快速跑
    int M = 512;
    int N = 512;
    int K = 512;
    int iters = 50;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        iters = std::atoi(argv[4]);
    }

    std::cout << "[naive_gemm] M=" << M << " N=" << N << " K=" << K
              << " iters=" << iters << "\n";

    // 初始化输入
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

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 1) 正确性验证
    naiveGemmKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost));
    cpuGemmRef(hA, hB, hRef, M, N, K);

    float err = maxAbsError(hC, hRef);
    std::cout << std::fixed << std::setprecision(6)
              << "[correctness] max_abs_err = " << err << "\n";

    // 2) 性能测试
    // warmup
    for (int i = 0; i < 10; ++i) {
        naiveGemmKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        naiveGemmKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    // GEMM FLOPs: 2 * M * N * K
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    std::cout << "[perf] avg_kernel_ms = " << avg_ms
              << ", throughput = " << gflops << " GFLOPS\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
