// bw_pcie.cu -- host-to-device (and D2H) bandwidth with pinned host memory.
// Copies 1 GiB per trial; 5 timed trials; prints every trial so the caller can
// take median and max. Peak allocation: 1 GiB pinned + 1 GiB device.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
  exit(1);} } while(0)

int main(int argc, char** argv) {
    const size_t bytes = 1ull << 30;          // 1 GiB >= 1 GB requirement
    const int trials = (argc > 1) ? atoi(argv[1]) : 5;

    void *h = nullptr, *d = nullptr;
    CK(cudaMallocHost(&h, bytes));            // pinned
    CK(cudaMalloc(&d, bytes));
    memset(h, 1, bytes);

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    // Warm-up (not timed): first transfer pays page-table / context costs.
    CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());

    printf("# bytes_per_copy %zu\n", bytes);
    for (int dir = 0; dir < 2; ++dir) {
        const char* name = dir == 0 ? "H2D" : "D2H";
        for (int t = 0; t < trials; ++t) {
            CK(cudaEventRecord(t0));
            if (dir == 0) CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
            else          CK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            float ms = 0.f;
            CK(cudaEventElapsedTime(&ms, t0, t1));
            printf("%s trial %d  %8.3f ms  %7.2f GB/s\n",
                   name, t, ms, bytes / (ms * 1e-3) / 1e9);
        }
    }

    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
    CK(cudaFree(d)); CK(cudaFreeHost(h));
    CK(cudaDeviceReset());                    // free everything
    return 0;
}
