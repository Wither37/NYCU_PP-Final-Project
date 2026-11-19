// Shared helpers: CUDA error checking, timing, the kernel function.
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                     \
                   cudaGetErrorName(err_), __FILE__, __LINE__,                 \
                   cudaGetErrorString(err_));                                  \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

// Launch-error check; synchronous checking only in debug builds.
#ifdef KKM_DEBUG_SYNC
#define CUDA_CHECK_LAUNCH()                \
  do {                                     \
    CUDA_CHECK(cudaGetLastError());        \
    CUDA_CHECK(cudaDeviceSynchronize());   \
  } while (0)
#else
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
#endif

namespace kkm {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

// Largest k the shared-memory accumulators support.
constexpr int kMaxK = 1024;

// Polynomial kernel (x^T y + 1)^2, matching Popcorn. Every strategy and the
// CPU reference go through this one definition.
template <typename T>
__host__ __device__ inline T kernel_from_dot(T dot) {
  T t = dot + T(1);
  return t * t;
}

class WallTimer {
 public:
  WallTimer() : t0_(std::chrono::steady_clock::now()) {}
  double ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point t0_;
};

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

}  // namespace kkm
