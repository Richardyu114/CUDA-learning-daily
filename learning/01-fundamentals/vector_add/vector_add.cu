#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/benchmark.hpp"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

void vectorAddCPU(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    int n = 1 << 24; // ~16M elements
    if (argc > 1) n = std::atoi(argv[1]);

    std::vector<float> h_a(n), h_b(n), h_c_gpu(n), h_c_cpu(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i % 1000) * 0.001f;
        h_b[i] = static_cast<float>((i * 7) % 1000) * 0.001f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // Correctness run
    vectorAddKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    vectorAddCPU(h_a, h_b, h_c_cpu);

    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(h_c_gpu[i] - h_c_cpu[i]));
    }

    // Benchmark CPU
    double cpu_ms = benchmark_ms([&]() { vectorAddCPU(h_a, h_b, h_c_cpu); }, 1, 3);

    // Benchmark GPU kernel-only (data already on device)
    double gpu_kernel_ms = benchmark_ms([&]() {
        vectorAddKernel<<<grid, block>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }, 3, 20);

    // Throughput (read A+B + write C = 3 * n * sizeof(float))
    double bytes = 3.0 * n * sizeof(float);
    double gb_s = (bytes / (gpu_kernel_ms / 1000.0)) / 1e9;

    std::cout << "[VectorAdd] n=" << n << " block=" << block << " grid=" << grid << "\n";
    std::cout << "max_abs_err: " << max_abs_err << "\n";
    std::cout << "cpu_ms(avg): " << cpu_ms << "\n";
    std::cout << "gpu_kernel_ms(avg): " << gpu_kernel_ms << "\n";
    std::cout << "gpu_effective_bw(GB/s): " << gb_s << "\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
