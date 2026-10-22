// devinfo.cu -- print cudaGetDeviceProperties for the selected device.
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
  return 1;} } while(0)

// FP32 cores per SM, by compute capability.
static int coresPerSM(int major, int minor) {
    switch (major * 10 + minor) {
        case 60: return 64;   // Pascal GP100
        case 61: case 62: return 128;
        case 70: case 72: case 75: return 64;
        case 80: return 64;   // Ampere A100
        case 86: case 87: case 89: return 128; // Ampere GA10x / Ada
        case 90: return 128;
        default: return -1;
    }
}

int main() {
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    printf("visible devices: %d\n", n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, i));
        int cps = coresPerSM(p.major, p.minor);
        double boostGHz = p.clockRate / 1e6;
        printf("--- device %d ---\n", i);
        printf("name                 : %s\n", p.name);
        printf("compute capability   : %d.%d\n", p.major, p.minor);
        printf("SM count             : %d\n", p.multiProcessorCount);
        printf("FP32 cores per SM    : %d\n", cps);
        printf("boost clock          : %.3f GHz\n", boostGHz);
        printf("memory clock         : %.0f MHz\n", p.memoryClockRate / 1e3);
        printf("memory bus width     : %d bit\n", p.memoryBusWidth);
        printf("total global memory  : %.0f MiB\n", p.totalGlobalMem / 1048576.0);
        printf("L2 cache             : %d KiB\n", p.l2CacheSize / 1024);
        printf("shared mem per block : %zu B\n", p.sharedMemPerBlock);
        printf("shared mem per SM    : %zu B\n", p.sharedMemPerMultiprocessor);
        printf("warp size            : %d\n", p.warpSize);
        printf("async engine count   : %d\n", p.asyncEngineCount);
        printf("ECC enabled          : %d\n", p.ECCEnabled);
        printf("theoretical FP32 peak: %.2f TFLOP/s  (SM x cores x 2 x clock)\n",
               p.multiProcessorCount * (double)cps * 2.0 * boostGHz / 1000.0);
        printf("theoretical mem BW   : %.1f GB/s  (2 x memclk x bus/8)\n",
               2.0 * (p.memoryClockRate * 1e3) * (p.memoryBusWidth / 8.0) / 1e9);
    }
    return 0;
}
