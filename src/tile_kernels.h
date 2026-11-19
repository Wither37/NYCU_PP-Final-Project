// Building blocks shared by the strategies that materialise K (in-memory, S1,
// S2): build a rectangular block of K from X, and accumulate a block of K into
// S by cluster label. All matrices are fp32 row-major.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace kkm {

// out[i*ldo + j] = kernel(X[r0+i] . X[c0+j])   for i < R, j < C.
// Tiled shared-memory product (hand-written; no cuBLAS).
void launch_build_tile(const float* d_X, int d, int64_t r0, int64_t c0, int R,
                       int C, float* out, size_t ldo, cudaStream_t stream);

// diag[i] = kernel(X[i] . X[i]).
void launch_diag(const float* d_X, int64_t n, int d, float* diag,
                 cudaStream_t stream);

// Row contribution of a block B (R x C, leading dimension ldb) whose columns
// are points with labels col_labels[0..C):
//   S_rows[i*k + col_labels[l]] (+)= B[i*ldb + l]
// With atomic_out == false the block owns its rows of S and overwrites them;
// with atomic_out == true it adds atomically (other blocks of K also write
// these rows, possibly from concurrent streams).
void launch_accum_rows(const float* B, size_t ldb, int R, int C,
                       const int* col_labels, float* S_rows, int k,
                       bool atomic_out, cudaStream_t stream);

// Column (transposed) contribution of the same block, used for the mirrored
// half of a symmetric K when only upper-triangular blocks are stored:
//   S_cols[l*k + row_labels[i]] += B[i*ldb + l]     (always atomic)
void launch_accum_cols(const float* B, size_t ldb, int R, int C,
                       const int* row_labels, float* S_cols, int k,
                       cudaStream_t stream);

}  // namespace kkm
