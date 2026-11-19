// OpenMP CPU implementation: the correctness reference for every GPU strategy
// and the CPU baseline for speedups. Accumulates in double.
#pragma once

#include <cstdint>
#include <vector>

#include "data.h"

namespace kkm {

struct IterStats {
  double objective = 0;  // sum_i min_j D[i][j] after the assignment
  int64_t changed = 0;   // points whose label changed
  int nonempty = 0;      // clusters with |L_j| > 0 entering the iteration
};

// Shared assignment math (Eqs. 2-4 of the proposal), double precision.
// Empty clusters are excluded from the argmin; ties go to the smallest j.
IterStats cpu_assign(const double* S, const double* diag, const int* labels_old,
                     int* labels_new, int64_t n, int k);

// diag[i] = K[i][i].
std::vector<double> cpu_diag(const Dataset& ds);

// S[i][j] = sum_{l: c(l) = j} K[i][l] with K evaluated on the fly in double.
// O(n^2 d) per call, no n^2 storage; used by tests as an independent oracle.
void cpu_compute_S_direct(const Dataset& ds, const int* labels, int k, double* S);

class CpuKKMeans {
 public:
  // Throws BudgetError if the n x n fp32 kernel matrix exceeds host_budget.
  CpuKKMeans(const Dataset& ds, int k, size_t host_budget_bytes);

  void build_K();  // n x n fp32, stored in full like Popcorn's CPU baseline
  void compute_S(const int* labels, double* S) const;  // n x k

  const std::vector<double>& diag() const { return diag_; }
  int64_t n() const { return n_; }
  int k() const { return k_; }

 private:
  const Dataset& ds_;
  int64_t n_;
  int k_;
  std::vector<float> K_;
  std::vector<double> diag_;
};

struct CpuRunResult {
  std::vector<int> labels;
  std::vector<IterStats> iters;
  std::vector<double> iter_ms;
  double build_ms = 0;
};

CpuRunResult cpu_run(const Dataset& ds, int k, int iters,
                     const std::vector<int>& init_labels,
                     size_t host_budget_bytes);

}  // namespace kkm
