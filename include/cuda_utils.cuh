#pragma once

/* GPU error-checking helpers. Only usable from CUDA (.cu) translation units.

   cudaSafeCall(expr)  - wrap any CUDA runtime call that returns cudaError_t
   cudaCheckError()    - call after a kernel launch to catch launch errors
                          and (in debug builds) synchronously catch execution
                          errors via a following cudaDeviceSynchronize()

   Both are no-ops if CUDA_ERROR_CHECK is undefined, so they can be compiled
   out entirely for release/benchmark builds where the extra
   cudaDeviceSynchronize() per kernel would distort timing. */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#ifndef CUDA_ERROR_CHECK
#define CUDA_ERROR_CHECK
#endif

#define cudaSafeCall(err) ::solar_gpu::detail::cuda_safe_call_impl((err), __FILE__, __LINE__)
#define cudaCheckError()  ::solar_gpu::detail::cuda_check_error_impl(__FILE__, __LINE__)

namespace solar_gpu::detail {

inline void cuda_safe_call_impl(cudaError_t err, const char* file, int line) {
#ifdef CUDA_ERROR_CHECK
    if (err != cudaSuccess) {
        std::cerr << "cudaSafeCall() failed at " << file << ":" << line
                   << " : " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
#endif
}

inline void cuda_check_error_impl(const char* file, int line) {
#ifdef CUDA_ERROR_CHECK
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cudaCheckError() failed at " << file << ":" << line
                   << " : " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaCheckError() with sync failed at " << file << ":" << line
                   << " : " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
#endif
}

} // namespace solar_gpu::detail
