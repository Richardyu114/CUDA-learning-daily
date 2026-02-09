#include <cuda_runtime.h>

#include <algorithm>
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

__global__ void saxpyKernel(float alpha, const float* x, const float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = alpha * x[idx] + y[idx];
}

void saxpyCPU(float alpha, const std::vector<float>& x, const std::vector<float>& y, std::vector<float>& out) {
    for (size_t i = 0; i < x.size(); ++i) out[i] = alpha * x[i] + y[i];
}

int main(int argc, char** argv) {
    int n = 1 << 24;
    if (argc > 1) n = std::atoi(argv[1]);
    float alpha = 2.5f;

    std::vector<float> h_x(n), h_y(n), h_out_gpu(n), h_out_cpu(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i % 1000) * 0.001f;
        h_y[i] = static_cast<float>((i * 3) % 1000) * 0.001f;
    }

    float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n + block - 1) / block;

    saxpyKernel<<<grid, block>>>(alpha, d_x, d_y, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    saxpyCPU(alpha, h_x, h_y, h_out_cpu);

    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        max_abs_err = std::max(max_abs_err, static_cast<double>(std::abs(h_out_gpu[i] - h_out_cpu[i])));
    }

    double cpu_ms = benchmark_ms([&]() { saxpyCPU(alpha, h_x, h_y, h_out_cpu); }, 1, 3);
    double gpu_ms = benchmark_ms([&]() {
        saxpyKernel<<<grid, block>>>(alpha, d_x, d_y, d_out, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }, 3, 20);

    // read x+y, write out = 3 arrays
    double bytes = 3.0 * n * sizeof(float);
    double gb_s = (bytes / (gpu_ms / 1000.0)) / 1e9;

    std::cout << "[SAXPY] n=" << n << " alpha=" << alpha << " block=" << block << "\n";
    std::cout << "max_abs_err: " << max_abs_err << "\n";
    std::cout << "cpu_ms(avg): " << cpu_ms << "\n";
    std::cout << "gpu_kernel_ms(avg): " << gpu_ms << "\n";
    std::cout << "gpu_effective_bw(GB/s): " << gb_s << "\n";

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
