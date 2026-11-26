#include <algorithm>
#include <cfloat>

#include "assign.h"
#include "common.h"

namespace kkm {

namespace {

// |L_j| and sum_{i in L_j} S[i][j], reduced per block in shared memory first.
__global__ void cluster_stats_kernel(const float* __restrict__ S,
                                     const int* __restrict__ labels, int64_t n,
                                     int k, int* __restrict__ size,
                                     double* __restrict__ normsum) {
  extern __shared__ unsigned char smem_stats[];
  double* sum = reinterpret_cast<double*>(smem_stats);
  int* cnt = reinterpret_cast<int*>(sum + k);
  for (int j = threadIdx.x; j < k; j += blockDim.x) {
    sum[j] = 0.0;
    cnt[j] = 0;
  }
  __syncthreads();
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int c = labels[i];
    atomicAdd(&cnt[c], 1);
    atomicAdd(&sum[c], static_cast<double>(S[i * k + c]));
  }
  __syncthreads();
  for (int j = threadIdx.x; j < k; j += blockDim.x) {
    if (cnt[j]) {
      atomicAdd(&size[j], cnt[j]);
      atomicAdd(&normsum[j], sum[j]);
    }
  }
}

__global__ void finalize_kernel(const int* __restrict__ size,
                                const double* __restrict__ normsum, int k,
                                double* __restrict__ inv, double* __restrict__ cnorm,
                                int* __restrict__ nonempty) {
  for (int j = threadIdx.x; j < k; j += blockDim.x) {
    if (size[j] > 0) {
      const double iv = 1.0 / size[j];
      inv[j] = iv;
      cnorm[j] = normsum[j] * iv * iv;
      atomicAdd(nonempty, 1);
    } else {
      inv[j] = 0.0;  // marks the cluster empty
      cnorm[j] = 0.0;
    }
  }
}

constexpr int kArgminThreads = 256;
constexpr int kArgminWarps = kArgminThreads / 32;

// One warp per point; lanes scan j, then a shuffle reduction keeps the
// smallest D and, on ties, the smallest j.
__global__ void __launch_bounds__(kArgminThreads)
argmin_kernel(const float* __restrict__ S, const float* __restrict__ diag,
              const int* __restrict__ labels_old, int* __restrict__ labels_new,
              const double* __restrict__ inv, const double* __restrict__ cnorm,
              int64_t n, int k, double* __restrict__ objective,
              unsigned long long* __restrict__ changed) {
  __shared__ double obj_s[kArgminWarps];
  __shared__ unsigned long long chg_s[kArgminWarps];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  double obj = 0.0;
  unsigned long long chg = 0;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kArgminWarps + warp; i < n;
       i += static_cast<int64_t>(gridDim.x) * kArgminWarps) {
    double best = DBL_MAX;
    int arg = INT_MAX;
    const double di = diag[i];
    for (int j = lane; j < k; j += 32) {
      const double iv = inv[j];
      if (iv == 0.0) continue;  // empty cluster
      const double D = di - 2.0 * static_cast<double>(S[i * k + j]) * iv + cnorm[j];
      if (D < best) {  // j increases per lane, so strict keeps the smaller j
        best = D;
        arg = j;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const double ob = __shfl_down_sync(0xffffffffu, best, off);
      const int oa = __shfl_down_sync(0xffffffffu, arg, off);
      if (ob < best || (ob == best && oa < arg)) {
        best = ob;
        arg = oa;
      }
    }
    if (lane == 0) {
      labels_new[i] = arg;
      obj += best;
      chg += (arg != labels_old[i]);
    }
  }
  if (lane == 0) {
    obj_s[warp] = obj;
    chg_s[warp] = chg;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double o = 0.0;
    unsigned long long c = 0;
    for (int w = 0; w < kArgminWarps; ++w) {
      o += obj_s[w];
      c += chg_s[w];
    }
    atomicAdd(objective, o);
    atomicAdd(changed, c);
  }
}

}  // namespace

GpuAssign::GpuAssign(int64_t n, int k, DeviceBudget& budget)
    : n_(n), k_(k), budget_(budget) {
  if (k < 1 || k > kMaxK) throw std::invalid_argument("k must be in [1, 1024]");
  size_ = budget_.alloc_n<int>(k, "assign.size");
  normsum_ = budget_.alloc_n<double>(k, "assign.normsum");
  inv_ = budget_.alloc_n<double>(k, "assign.inv");
  cnorm_ = budget_.alloc_n<double>(k, "assign.cnorm");
  objective_ = budget_.alloc_n<double>(1, "assign.objective");
  changed_ = budget_.alloc_n<unsigned long long>(1, "assign.changed");
  nonempty_ = budget_.alloc_n<int>(1, "assign.nonempty");
}

GpuAssign::~GpuAssign() {
  for (void* p : {static_cast<void*>(size_), static_cast<void*>(normsum_),
                  static_cast<void*>(inv_), static_cast<void*>(cnorm_),
                  static_cast<void*>(objective_), static_cast<void*>(changed_),
                  static_cast<void*>(nonempty_)})
    budget_.free(p);
}

IterStats GpuAssign::run(const float* S, const float* diag, const int* labels_old,
                         int* labels_new, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(size_, 0, k_ * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(normsum_, 0, k_ * sizeof(double), stream));
  CUDA_CHECK(cudaMemsetAsync(objective_, 0, sizeof(double), stream));
  CUDA_CHECK(cudaMemsetAsync(changed_, 0, sizeof(unsigned long long), stream));
  CUDA_CHECK(cudaMemsetAsync(nonempty_, 0, sizeof(int), stream));

  const int threads = 256;
  const int64_t sblocks = std::min<int64_t>(ceil_div(n_, threads), 1024);
  const size_t smem = static_cast<size_t>(k_) * (sizeof(double) + sizeof(int));
  cluster_stats_kernel<<<static_cast<unsigned>(sblocks), threads, smem, stream>>>(
      S, labels_old, n_, k_, size_, normsum_);
  CUDA_CHECK_LAUNCH();
  finalize_kernel<<<1, 1024, 0, stream>>>(size_, normsum_, k_, inv_, cnorm_,
                                          nonempty_);
  CUDA_CHECK_LAUNCH();
  const int64_t ablocks = std::min<int64_t>(ceil_div(n_, kArgminWarps), 1 << 16);
  argmin_kernel<<<static_cast<unsigned>(ablocks), kArgminThreads, 0, stream>>>(
      S, diag, labels_old, labels_new, inv_, cnorm_, n_, k_, objective_, changed_);
  CUDA_CHECK_LAUNCH();

  IterStats st;
  unsigned long long chg = 0;
  CUDA_CHECK(cudaMemcpyAsync(&st.objective, objective_, sizeof(double),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&chg, changed_, sizeof chg, cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(&st.nonempty, nonempty_, sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  st.changed = static_cast<int64_t>(chg);
  return st;
}

}  // namespace kkm
