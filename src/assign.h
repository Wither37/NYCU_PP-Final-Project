// Shared assignment module (Eqs. 2-4 of the proposal), identical for every
// GPU strategy: centroid norms, D, row-wise argmin, cluster sizes. Touches
// only O(nk) data.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "budget.h"
#include "cpu_ref.h"  // IterStats

namespace kkm {

class GpuAssign {
 public:
  GpuAssign(int64_t n, int k, DeviceBudget& budget);
  ~GpuAssign();
  GpuAssign(const GpuAssign&) = delete;
  GpuAssign& operator=(const GpuAssign&) = delete;

  // S: n x k fp32, diag: n, labels_old -> labels_new. Same rules as
  // cpu_assign: empty clusters excluded, ties to the smallest j. D is
  // evaluated in double. Synchronises `stream` to return the statistics.
  IterStats run(const float* S, const float* diag, const int* labels_old,
                int* labels_new, cudaStream_t stream);

 private:
  int64_t n_;
  int k_;
  DeviceBudget& budget_;
  // Device workspace.
  int* size_ = nullptr;
  double* normsum_ = nullptr;
  double* inv_ = nullptr;
  double* cnorm_ = nullptr;
  double* objective_ = nullptr;
  unsigned long long* changed_ = nullptr;
  int* nonempty_ = nullptr;
};

}  // namespace kkm
