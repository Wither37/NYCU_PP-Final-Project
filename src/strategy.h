// The one interface every strategy implements. All strategies compute the same
// S (n x k, row-major fp32):  S[i][j] = sum_{l : c(l) = j} K[i][l]
// and differ only in where K lives. The shared assignment module consumes S.
//
//   inmem  full K resident on the GPU (Popcorn's layout)        4 n^2 bytes
//   s1     upper-triangular tiles resident on the GPU            ~2 n^2 bytes
//   s2     upper-triangular tiles in pinned host memory, streamed over PCIe
//   s3     fused recomputation, no K at all (owned by S3's author)
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "budget.h"

namespace kkm {

struct StrategyOptions {
  int tile = 4096;  // tile edge for s1/s2
  int nbuf = 3;     // device tile buffers (= streams) for s2
};

struct StrategyContext {
  int64_t n = 0;
  int d = 0;
  int k = 0;
  const float* d_X = nullptr;  // n x d on the device, owned by the driver
  DeviceBudget* dev = nullptr;
  HostBudget* host = nullptr;
};

class SStrategy {
 public:
  virtual ~SStrategy() = default;
  virtual std::string name() const = 0;

  // One-time construction of whatever representation of K the strategy uses.
  // All device memory goes through ctx.dev, all pinned memory through
  // ctx.host; either may throw BudgetError. Returns when construction is done.
  virtual void setup(const StrategyContext& ctx) = 0;

  // Fully overwrites d_S. Work may run on internal streams, but must be
  // complete with respect to `stream`: anything enqueued on `stream`
  // afterwards sees the final S.
  virtual void compute_S(const int* d_labels, float* d_S, cudaStream_t stream) = 0;

  // Extra key/value pairs for the run report (as a JSON object body,
  // e.g. "\"tiles\": 10, \"pinned_bytes\": 123").
  virtual std::string report_json() const { return ""; }
};

// "inmem", "s1", "s2". Throws std::invalid_argument for unknown names.
std::unique_ptr<SStrategy> make_strategy(const std::string& name,
                                         const StrategyOptions& opts);

// Upper-triangular tiling of an n x n symmetric matrix with tile edge T:
// tiles (I, J) with I <= J, row-major order. Diagonal tiles are stored in
// full (both triangles) and contribute only their rows; off-diagonal tiles
// contribute rows (to S rows of I) and columns (to S rows of J).
struct Tile {
  int I = 0, J = 0;
  int64_t r0 = 0, c0 = 0;
  int R = 0, C = 0;
  size_t elems() const { return static_cast<size_t>(R) * C; }
};
std::vector<Tile> upper_tiles(int64_t n, int T);

}  // namespace kkm
