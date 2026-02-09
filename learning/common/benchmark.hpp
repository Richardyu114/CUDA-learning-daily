#pragma once

#include <chrono>
#include <functional>
#include <string>

inline double benchmark_ms(const std::function<void()>& fn, int warmup = 3, int iters = 10) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t1 - t0;
    return ms.count() / iters;
}
