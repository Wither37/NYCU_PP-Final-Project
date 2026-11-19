// Drives a full clustering run on the GPU with one S strategy and the shared
// assignment module.
#pragma once

#include <string>
#include <vector>

#include "cpu_ref.h"
#include "data.h"
#include "strategy.h"

namespace kkm {

struct GpuRunOptions {
  std::string strategy = "s2";
  int k = 10;
  int iters = 30;
  size_t device_budget = size_t(24) << 30;  // nominal; 10% headroom applied
  size_t host_budget = 0;                   // 0 -> 0.8 * MemAvailable
  StrategyOptions sopts;
};

struct GpuRunResult {
  std::vector<int> labels;
  std::vector<IterStats> iters;
  std::vector<double> iter_ms;    // wall time per iteration (S + assign)
  std::vector<double> s_ms;       // compute_S per iteration (CUDA events)
  double upload_ms = 0;           // X, labels to the device, diag
  double setup_ms = 0;            // strategy construction (e.g. building K)
  size_t device_limit = 0, device_peak = 0;
  size_t host_limit = 0, host_peak = 0;
  std::string strategy_json;      // SStrategy::report_json()
};

// Throws BudgetError when the configuration does not fit the budgets.
GpuRunResult gpu_run(const Dataset& ds, const std::vector<int>& init_labels,
                     const GpuRunOptions& opt);

// Computes S once for the given labels with the named strategy and returns it
// (n x k). For tests.
std::vector<float> gpu_compute_S_once(const Dataset& ds, const std::vector<int>& labels,
                                      int k, const std::string& strategy,
                                      const StrategyOptions& sopts,
                                      size_t device_budget, size_t host_budget);

}  // namespace kkm
