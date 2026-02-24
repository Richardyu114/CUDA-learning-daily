// Lesson 01: Vector Add — 理解 GPU 执行模型
// 
// 目标：
//   1. 理解 host vs device 的概念
//   2. 理解 grid → block → thread 的层次结构
//   3. 理解 memory 搬运（cudaMemcpy）的开销
//   4. 学会用 CUDA event 做 timing
//
// 编译：nvcc -o 01_vector_add 01_vector_add.cu
// 运行：./01_vector_add
//
// 作者：Camus × Richard

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ============================================================
// GPU Kernel：每个 thread 负责一个元素的加法
// ============================================================
__global__ void vector_add_v1(const float *a, const float *b, float *c, int n) {
    // blockIdx.x  — 当前 thread 所在的 block 编号
    // blockDim.x  — 每个 block 有多少个 thread
    // threadIdx.x — 当前 thread 在 block 内的编号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================================
// GPU Kernel V2：每个 thread 处理多个元素（grid-stride loop）
// 
// 为什么需要这个？
// 当 n 很大时，我们不一定要启动 n 个 thread。
// 用更少的 thread + 循环，可以提高 occupancy 和灵活性。
// 这也是 vLLM kernel 中常见的写法。
// ============================================================
__global__ void vector_add_v2(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // 总 thread 数
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================
// CPU 参考实现
// ============================================================
void vector_add_cpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================
// 验证结果
// ============================================================
bool verify(const float *ref, const float *test, int n, float tol = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(ref[i] - test[i]) > tol) {
            printf("Mismatch at index %d: ref=%.6f, test=%.6f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

// ============================================================
// 计时辅助宏
// ============================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    // --- 参数 ---
    const int N = 1 << 24;  // 16M 元素
    const size_t bytes = N * sizeof(float);

    printf("=== Lesson 01: Vector Add ===\n");
    printf("N = %d (%.1f MB per array)\n\n", N, bytes / 1e6);

    // --- Host 内存分配 ---
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c_cpu = (float*)malloc(bytes);
    float *h_c_gpu = (float*)malloc(bytes);

    // 初始化
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 1000) / 100.0f;
        h_b[i] = (float)(rand() % 1000) / 100.0f;
    }

    // --- CPU 计算（参考） ---
    vector_add_cpu(h_a, h_b, h_c_cpu, N);

    // --- Device 内存分配 ---
    // 
    // 思考题：这里的 cudaMalloc 分配的是 GPU 显存（device memory）。
    // CPU 不能直接读写这块内存，必须通过 cudaMemcpy 搬运。
    // 这个搬运走的是 PCIe 总线，速度远慢于 GPU 内部带宽。
    //
    // 在 vLLM 中，这就是为什么 KV cache 管理如此重要——
    // 尽量避免不必要的 host-device 搬运。
    //
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // --- H2D 搬运 ---
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // --- 计时用 CUDA Event ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // =====================
    // V1: 简单版
    // =====================
    int threads_per_block = 256;  // 常见选择：128 / 256 / 512
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    printf("[V1] threads_per_block=%d, blocks=%d, total_threads=%d\n",
           threads_per_block, blocks, threads_per_block * blocks);

    CUDA_CHECK(cudaEventRecord(start));
    vector_add_v1<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_v1 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_v1, start, stop));

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("[V1] Time: %.3f ms | Correct: %s\n\n",
           ms_v1, verify(h_c_cpu, h_c_gpu, N) ? "YES ✅" : "NO ❌");

    // =====================
    // V2: Grid-stride loop
    // =====================
    // 故意用少一些的 block，让每个 thread 做更多工作
    int blocks_v2 = 256;  // 远少于 V1 的 blocks

    printf("[V2] threads_per_block=%d, blocks=%d, total_threads=%d\n",
           threads_per_block, blocks_v2, threads_per_block * blocks_v2);

    CUDA_CHECK(cudaEventRecord(start));
    vector_add_v2<<<blocks_v2, threads_per_block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_v2 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_v2, start, stop));

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("[V2] Time: %.3f ms | Correct: %s\n\n",
           ms_v2, verify(h_c_cpu, h_c_gpu, N) ? "YES ✅" : "NO ❌");

    // =====================
    // 思考题输出
    // =====================
    printf("=== 思考题 ===\n");
    printf("1. V1 和 V2 的性能差距大吗？为什么？\n");
    printf("   提示：vector_add 是 memory-bound 的，瓶颈在显存带宽，不在计算。\n");
    printf("   两个版本的内存访问模式几乎一样（都是 coalesced），所以差距不大。\n\n");
    printf("2. 试试改 threads_per_block 为 64 / 128 / 512 / 1024，观察性能变化。\n");
    printf("   哪个最快？为什么？（提示：与 SM 的 warp scheduler 和 occupancy 有关）\n\n");
    printf("3. 如果把 N 改成 1<<28 (256M)，H2D memcpy 的时间占比会怎样变化？\n");
    printf("   在真实系统中（如 vLLM），这就是为什么要用 pinned memory 和 async copy。\n\n");

    // --- 清理 ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    printf("Done! 🎉\n");
    return 0;
}
