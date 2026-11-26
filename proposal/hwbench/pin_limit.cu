// How much host memory can be pinned with cudaHostAlloc? Allocates chunks
// until failure or a cap, timing each, then frees everything at once.
//   ./pin_limit [chunk_GiB=8] [cap_GiB=160]
// Pinned memory cannot be swapped: keep the cap well below MemAvailable on
// the shared host.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  const double chunk_gib = argc > 1 ? atof(argv[1]) : 8.0;
  const double cap_gib = argc > 2 ? atof(argv[2]) : 160.0;
  const size_t chunk = static_cast<size_t>(chunk_gib * (1ull << 30));
  std::vector<void*> ptrs;
  double total = 0;
  printf("# chunk %.1f GiB, cap %.1f GiB\n", chunk_gib, cap_gib);
  while (total + chunk_gib <= cap_gib + 1e-9) {
    void* p = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    cudaError_t e = cudaHostAlloc(&p, chunk, cudaHostAllocDefault);
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
    if (e != cudaSuccess) {
      printf("FAIL after %.1f GiB: %s (%.1f ms)\n", total, cudaGetErrorString(e), ms);
      cudaGetLastError();
      break;
    }
    ptrs.push_back(p);
    total += chunk_gib;
    printf("pinned %7.1f GiB  (+%.1f GiB in %8.1f ms, %.2f GiB/s)\n", total,
           chunk_gib, ms, chunk_gib / (ms / 1000.0));
    fflush(stdout);
  }
  auto t0 = std::chrono::steady_clock::now();
  for (void* p : ptrs) cudaFreeHost(p);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
  printf("freed %.1f GiB in %.1f ms\n", total, ms);
  cudaDeviceReset();
  return 0;
}
