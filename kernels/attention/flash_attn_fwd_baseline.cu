#include <cuda_runtime.h>

#include <cmath>
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

// Baseline educational kernel: one thread computes one (b,h,i,d) output using full softmax over j.
// Layout: [B, H, S, D]
__global__ void flashAttnFwdBaseline(const float* Q, const float* K, const float* V,
                                     float* O, int B, int H, int S, int D, bool causal) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int bh = blockIdx.z;
    if (d >= D || i >= S) return;

    int b = bh / H;
    int h = bh % H;

    auto idx = [&](int bb, int hh, int ss, int dd) { return ((bb * H + hh) * S + ss) * D + dd; };

    float max_logit = -1e30f;
    for (int j = 0; j < S; ++j) {
        if (causal && j > i) break;
        float dot = 0.0f;
        for (int t = 0; t < D; ++t) dot += Q[idx(b, h, i, t)] * K[idx(b, h, j, t)];
        float logit = dot / sqrtf((float)D);
        max_logit = fmaxf(max_logit, logit);
    }

    float denom = 0.0f;
    float numer = 0.0f;
    for (int j = 0; j < S; ++j) {
        if (causal && j > i) break;
        float dot = 0.0f;
        for (int t = 0; t < D; ++t) dot += Q[idx(b, h, i, t)] * K[idx(b, h, j, t)];
        float w = expf(dot / sqrtf((float)D) - max_logit);
        denom += w;
        numer += w * V[idx(b, h, j, d)];
    }

    O[idx(b, h, i, d)] = numer / fmaxf(denom, 1e-8f);
}

int main(int argc, char** argv) {
    int B = 1, H = 2, S = 64, D = 64;
    bool causal = true;
    if (argc >= 5) {
        B = std::atoi(argv[1]); H = std::atoi(argv[2]); S = std::atoi(argv[3]); D = std::atoi(argv[4]);
    }

    size_t elems = (size_t)B * H * S * D;
    std::vector<float> h_Q(elems), h_K(elems), h_V(elems), h_O(elems, 0.0f);
    for (size_t i = 0; i < elems; ++i) {
        h_Q[i] = ((i % 17) - 8) * 0.01f;
        h_K[i] = ((i % 19) - 9) * 0.01f;
        h_V[i] = ((i % 23) - 11) * 0.01f;
    }

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(128);
    dim3 grid((D + block.x - 1) / block.x, S, B * H);
    flashAttnFwdBaseline<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, S, D, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, elems * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[flash_attn_fwd_baseline] done. shape=" << B << "," << H << "," << S << "," << D
              << " causal=" << causal << "\n";
    std::cout << "O[0], O[mid], O[last] = " << h_O[0] << ", " << h_O[elems / 2] << ", " << h_O[elems - 1] << "\n";

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}
