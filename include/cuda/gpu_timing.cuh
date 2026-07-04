#pragma once

/* Small timing/memory-reporting helpers shared by both GPU pipeline
   instances. */

#include <chrono>
#include <cstdio>

#include <cuda_runtime.h>

namespace solar_gpu {

inline float print_gpu_mem(const char* label) {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    float used_mb = (float)((total_mem - free_mem) / 1048576.0);
    std::printf("  [GPU MEM] %s: %7.1f MB used\n", label, used_mb);
    return used_mb;
}

inline double elapsed_ms(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

} // namespace solar_gpu
