#include <algorithm>

#include "common.h"
#include "tile_kernels.h"

namespace kkm {

// --------------------------------------------------------------- build ---
//
// 64x64 output block per thread block, 256 threads, 4x4 outputs per thread,
// K-dimension (here: d) staged through shared memory 16 at a time.

namespace {

constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
constexpr int kBuildThreads = (BM / TM) * (BN / TN);  // 256

__global__ void __launch_bounds__(kBuildThreads)
build_tile_kernel(const float* __restrict__ X, int d, int64_t r0, int64_t c0,
                  int R, int C, float* __restrict__ out, size_t ldo) {
  __shared__ float As[BK][BM + 1];
  __shared__ float Bs[BK][BN + 1];

  const int tx = threadIdx.x % (BN / TN);
  const int ty = threadIdx.x / (BN / TN);
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;

  float acc[TM][TN];
#pragma unroll
  for (int a = 0; a < TM; ++a)
#pragma unroll
    for (int b = 0; b < TN; ++b) acc[a][b] = 0.f;

  for (int k0 = 0; k0 < d; k0 += BK) {
    // Consecutive threads read consecutive dims of one point: coalesced.
    for (int e = threadIdx.x; e < BM * BK; e += kBuildThreads) {
      const int m = e / BK, t = e % BK, gk = k0 + t;
      const int64_t gr = row0 + m, gc = col0 + m;
      As[t][m] = (gr < R && gk < d) ? X[(r0 + gr) * d + gk] : 0.f;
      Bs[t][m] = (gc < C && gk < d) ? X[(c0 + gc) * d + gk] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < BK; ++t) {
      float a[TM], b[TN];
#pragma unroll
      for (int q = 0; q < TM; ++q) a[q] = As[t][ty * TM + q];
#pragma unroll
      for (int q = 0; q < TN; ++q) b[q] = Bs[t][tx * TN + q];
#pragma unroll
      for (int p = 0; p < TM; ++p)
#pragma unroll
        for (int q = 0; q < TN; ++q) acc[p][q] = fmaf(a[p], b[q], acc[p][q]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int p = 0; p < TM; ++p) {
    const int64_t r = row0 + ty * TM + p;
    if (r >= R) continue;
#pragma unroll
    for (int q = 0; q < TN; ++q) {
      const int64_t c = col0 + tx * TN + q;
      if (c < C) out[r * ldo + c] = kernel_from_dot(acc[p][q]);
    }
  }
}

__global__ void diag_kernel(const float* __restrict__ X, int64_t n, int d,
                            float* __restrict__ diag) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float s = 0.f;
    for (int t = 0; t < d; ++t) s = fmaf(X[i * d + t], X[i * d + t], s);
    diag[i] = kernel_from_dot(s);
  }
}

// ------------------------------------------------------------ rows ---
//
// One warp per row of B. Each warp owns a k-bin accumulator in shared memory;
// lanes stream the row coalesced and add into bin col_labels[l]. Lanes that
// hit the same bin serialise on the shared atomic, which is cheap compared
// with the global-memory (or PCIe) cost of reading B.

constexpr int kRowsThreads = 256;
constexpr int kRowsWarps = kRowsThreads / 32;

template <bool kAtomicOut>
__global__ void __launch_bounds__(kRowsThreads)
accum_rows_kernel(const float* __restrict__ B, size_t ldb, int R, int C,
                  const int* __restrict__ col_labels, float* __restrict__ S_rows,
                  int k) {
  extern __shared__ float acc_rows[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  float* acc = acc_rows + warp * k;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kRowsWarps + warp; i < R;
       i += static_cast<int64_t>(gridDim.x) * kRowsWarps) {
    for (int j = lane; j < k; j += 32) acc[j] = 0.f;
    __syncwarp();

    const float* row = B + i * ldb;
    int l = lane;
    // Unrolled by 4 so four loads are in flight before the atomics.
    for (; l + 96 < C; l += 128) {
      const float v0 = row[l], v1 = row[l + 32], v2 = row[l + 64], v3 = row[l + 96];
      const int c0 = col_labels[l], c1 = col_labels[l + 32];
      const int c2 = col_labels[l + 64], c3 = col_labels[l + 96];
      atomicAdd(&acc[c0], v0);
      atomicAdd(&acc[c1], v1);
      atomicAdd(&acc[c2], v2);
      atomicAdd(&acc[c3], v3);
    }
    for (; l < C; l += 32) atomicAdd(&acc[col_labels[l]], row[l]);
    __syncwarp();

    float* out = S_rows + i * k;
    for (int j = lane; j < k; j += 32) {
      if (kAtomicOut) {
        if (acc[j] != 0.f) atomicAdd(&out[j], acc[j]);
      } else {
        out[j] = acc[j];
      }
    }
    __syncwarp();
  }
}

// ------------------------------------------------------------ cols ---
//
// Block = 32 columns x 8 row groups. Every lane of a warp reads the same row i,
// so all lanes add into the same label bin but different column slots: no
// intra-warp conflict. Accumulator is k x 32 floats in shared memory, flushed
// to global S with atomics (other row ranges and other blocks of K hit the
// same rows of S).

constexpr int kColsX = 32, kColsY = 8;
constexpr int kColsRowsPerBlock = 1024;
constexpr size_t kMaxSharedPerBlock = 48 * 1024;

__global__ void __launch_bounds__(kColsX* kColsY)
accum_cols_kernel(const float* __restrict__ B, size_t ldb, int R, int C,
                  const int* __restrict__ row_labels, float* __restrict__ S_cols,
                  int k) {
  extern __shared__ float acc_cols[];  // [k][32]
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * kColsX + tx;
  for (int e = tid; e < k * kColsX; e += kColsX * kColsY) acc_cols[e] = 0.f;
  __syncthreads();

  const int64_t l = static_cast<int64_t>(blockIdx.x) * kColsX + tx;
  const int64_t i0 = static_cast<int64_t>(blockIdx.y) * kColsRowsPerBlock;
  const int64_t i1 = min(static_cast<int64_t>(R), i0 + kColsRowsPerBlock);
  if (l < C) {
    int64_t i = i0 + ty;
    for (; i + 3 * kColsY < i1; i += 4 * kColsY) {
      const float v0 = B[i * ldb + l], v1 = B[(i + kColsY) * ldb + l];
      const float v2 = B[(i + 2 * kColsY) * ldb + l];
      const float v3 = B[(i + 3 * kColsY) * ldb + l];
      atomicAdd(&acc_cols[row_labels[i] * kColsX + tx], v0);
      atomicAdd(&acc_cols[row_labels[i + kColsY] * kColsX + tx], v1);
      atomicAdd(&acc_cols[row_labels[i + 2 * kColsY] * kColsX + tx], v2);
      atomicAdd(&acc_cols[row_labels[i + 3 * kColsY] * kColsX + tx], v3);
    }
    for (; i < i1; i += kColsY)
      atomicAdd(&acc_cols[row_labels[i] * kColsX + tx], B[i * ldb + l]);
  }
  __syncthreads();

  for (int e = tid; e < k * kColsX; e += kColsX * kColsY) {
    const int j = e / kColsX, c = e % kColsX;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * kColsX + c;
    const float v = acc_cols[e];
    if (col < C && v != 0.f) atomicAdd(&S_cols[col * k + j], v);
  }
}

// Fallback for k too large for the shared accumulator: direct global atomics.
__global__ void accum_cols_global_kernel(const float* __restrict__ B, size_t ldb,
                                         int R, int C,
                                         const int* __restrict__ row_labels,
                                         float* __restrict__ S_cols, int k) {
  const int64_t l = static_cast<int64_t>(blockIdx.x) * kColsX + threadIdx.x;
  const int64_t i0 = static_cast<int64_t>(blockIdx.y) * kColsRowsPerBlock;
  const int64_t i1 = min(static_cast<int64_t>(R), i0 + kColsRowsPerBlock);
  if (l >= C) return;
  for (int64_t i = i0 + threadIdx.y; i < i1; i += kColsY)
    atomicAdd(&S_cols[l * k + row_labels[i]], B[i * ldb + l]);
}

}  // namespace

void launch_build_tile(const float* d_X, int d, int64_t r0, int64_t c0, int R,
                       int C, float* out, size_t ldo, cudaStream_t stream) {
  if (R <= 0 || C <= 0) return;
  dim3 grid(static_cast<unsigned>(ceil_div(C, BN)),
            static_cast<unsigned>(ceil_div(R, BM)));
  if (grid.y > 65535) throw std::invalid_argument("build_tile: R too large");
  build_tile_kernel<<<grid, kBuildThreads, 0, stream>>>(d_X, d, r0, c0, R, C,
                                                        out, ldo);
  CUDA_CHECK_LAUNCH();
}

void launch_diag(const float* d_X, int64_t n, int d, float* diag,
                 cudaStream_t stream) {
  const int threads = 256;
  const int64_t blocks = std::min<int64_t>(ceil_div(n, threads), 65535);
  diag_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(d_X, n, d, diag);
  CUDA_CHECK_LAUNCH();
}

void launch_accum_rows(const float* B, size_t ldb, int R, int C,
                       const int* col_labels, float* S_rows, int k,
                       bool atomic_out, cudaStream_t stream) {
  if (R <= 0 || C <= 0) return;
  if (k > kMaxK) throw std::invalid_argument("accum_rows: k exceeds kMaxK");
  const int64_t blocks = std::min<int64_t>(ceil_div(R, kRowsWarps), 1 << 20);
  const size_t smem = static_cast<size_t>(kRowsWarps) * k * sizeof(float);
  if (atomic_out)
    accum_rows_kernel<true><<<static_cast<unsigned>(blocks), kRowsThreads, smem,
                              stream>>>(B, ldb, R, C, col_labels, S_rows, k);
  else
    accum_rows_kernel<false><<<static_cast<unsigned>(blocks), kRowsThreads, smem,
                               stream>>>(B, ldb, R, C, col_labels, S_rows, k);
  CUDA_CHECK_LAUNCH();
}

void launch_accum_cols(const float* B, size_t ldb, int R, int C,
                       const int* row_labels, float* S_cols, int k,
                       cudaStream_t stream) {
  if (R <= 0 || C <= 0) return;
  dim3 block(kColsX, kColsY);
  dim3 grid(static_cast<unsigned>(ceil_div(C, kColsX)),
            static_cast<unsigned>(ceil_div(R, kColsRowsPerBlock)));
  if (grid.y > 65535) throw std::invalid_argument("accum_cols: R too large");
  const size_t smem = static_cast<size_t>(k) * kColsX * sizeof(float);
  if (smem <= kMaxSharedPerBlock)
    accum_cols_kernel<<<grid, block, smem, stream>>>(B, ldb, R, C, row_labels,
                                                     S_cols, k);
  else
    accum_cols_global_kernel<<<grid, block, 0, stream>>>(B, ldb, R, C, row_labels,
                                                         S_cols, k);
  CUDA_CHECK_LAUNCH();
}

}  // namespace kkm
