#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
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

__global__ void reduceSumKernel(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

float reduceCPU(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0f);
}

float reduceGPU(const float* d_in, int n, int block) {
    int curr_n = n;
    float* d_curr_in = const_cast<float*>(d_in);
    float* d_out = nullptr;

    while (curr_n > 1) {
        int grid = (curr_n + block - 1) / block;
        CUDA_CHECK(cudaMalloc(&d_out, grid * sizeof(float)));

        reduceSumKernel<<<grid, block, block * sizeof(float)>>>(d_curr_in, d_out, curr_n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (d_curr_in != d_in) CUDA_CHECK(cudaFree(d_curr_in));
        d_curr_in = d_out;
        d_out = nullptr;
        curr_n = grid;
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_curr_in, sizeof(float), cudaMemcpyDeviceToHost));
    if (d_curr_in != d_in) CUDA_CHECK(cudaFree(d_curr_in));
    return result;
}

int main(int argc, char** argv) {
    int n = 1 << 24;
    if (argc > 1) n = std::atoi(argv[1]);
    int block = 256;

    std::vector<float> h_in(n);
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>((i % 100) * 0.01f);

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    float gpu_sum = reduceGPU(d_in, n, block);
    float cpu_sum = reduceCPU(h_in);
    double abs_err = std::abs(gpu_sum - cpu_sum);

    double cpu_ms = benchmark_ms([&]() {
        volatile float s = reduceCPU(h_in);
        (void)s;
    }, 1, 5);

    double gpu_ms = benchmark_ms([&]() {
        volatile float s = reduceGPU(d_in, n, block);
        (void)s;
    }, 2, 10);

    std::cout << "[ReduceSum] n=" << n << " block=" << block << "\n";
    std::cout << "cpu_sum: " << cpu_sum << " gpu_sum: " << gpu_sum << "\n";
    std::cout << "abs_err: " << abs_err << "\n";
    std::cout << "cpu_ms(avg): " << cpu_ms << "\n";
    std::cout << "gpu_ms(avg): " << gpu_ms << "\n";

    CUDA_CHECK(cudaFree(d_in));
    return 0;
}
