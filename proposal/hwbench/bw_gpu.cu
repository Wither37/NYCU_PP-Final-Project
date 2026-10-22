// bw_gpu.cu -- device memory bandwidth from a read+write copy kernel.
// Streams 1 GiB in and 1 GiB out per trial; effective bandwidth counts BOTH the
// read and the written bytes (2 x 1 GiB). 5 timed trials, one line each.
// Peak allocation: 2 GiB device (<= 4 GB budget).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
  exit(1);} } while(0)

// float4-wide grid-stride copy: one read + one write per element, no reuse,
// so it measures DRAM streaming bandwidth rather than cache bandwidth.
__global__ void copyKernel(const float4* __restrict__ src,
                           float4* __restrict__ dst, size_t n4) {
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
        dst[i] = src[i];
}

int main(int argc, char** argv) {
    const size_t bytes = 1ull << 30;          // per buffer; 1 GiB read + 1 GiB write
    const int trials = (argc > 1) ? atoi(argv[1]) : 5;
    const size_t n4 = bytes / sizeof(float4);

    float4 *src = nullptr, *dst = nullptr;
    CK(cudaMalloc(&src, bytes));
    CK(cudaMalloc(&dst, bytes));
    CK(cudaMemset(src, 1, bytes));

    const int threads = 256;
    int blocks = 0, dev = 0;
    cudaDeviceProp p;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&p, dev));
    // Saturate the device: a few waves of resident blocks.
    blocks = p.multiProcessorCount * 32;

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    copyKernel<<<blocks, threads>>>(src, dst, n4);   // warm-up
    CK(cudaDeviceSynchronize());

    printf("# bytes_moved_per_trial %zu (read+write), blocks %d threads %d\n",
           2 * bytes, blocks, threads);
    for (int t = 0; t < trials; ++t) {
        CK(cudaEventRecord(t0));
        copyKernel<<<blocks, threads>>>(src, dst, n4);
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        CK(cudaGetLastError());
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        printf("GPU trial %d  %8.3f ms  %7.2f GB/s\n",
               t, ms, 2.0 * bytes / (ms * 1e-3) / 1e9);
    }

    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
    CK(cudaFree(src)); CK(cudaFree(dst));
    CK(cudaDeviceReset());
    return 0;
}
