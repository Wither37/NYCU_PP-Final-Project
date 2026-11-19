// In-memory reference path: the full n x n K resident on the GPU, re-read
// every iteration (Popcorn's layout, but with our own hand-written kernels).
#include <sstream>

#include "common.h"
#include "strategy.h"
#include "tile_kernels.h"

namespace kkm {

namespace {

class InMemStrategy : public SStrategy {
 public:
  explicit InMemStrategy(const StrategyOptions&) {}
  ~InMemStrategy() override {
    if (dev_) dev_->free(K_);
  }

  std::string name() const override { return "inmem"; }

  void setup(const StrategyContext& ctx) override {
    n_ = ctx.n;
    k_ = ctx.k;
    dev_ = ctx.dev;
    K_ = dev_->alloc_n<float>(static_cast<size_t>(n_) * n_, "inmem.K");
    launch_build_tile(ctx.d_X, ctx.d, 0, 0, static_cast<int>(n_),
                      static_cast<int>(n_), K_, static_cast<size_t>(n_), 0);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void compute_S(const int* d_labels, float* d_S, cudaStream_t stream) override {
    // Every row of S is owned by exactly one warp: plain stores, no memset.
    launch_accum_rows(K_, static_cast<size_t>(n_), static_cast<int>(n_),
                      static_cast<int>(n_), d_labels, d_S, k_,
                      /*atomic_out=*/false, stream);
  }

  std::string report_json() const override {
    std::ostringstream o;
    o << "\"K_device_bytes\": " << static_cast<size_t>(n_) * n_ * sizeof(float);
    return o.str();
  }

 private:
  int64_t n_ = 0;
  int k_ = 0;
  DeviceBudget* dev_ = nullptr;
  float* K_ = nullptr;
};

}  // namespace

std::unique_ptr<SStrategy> make_inmem(const StrategyOptions& o) {
  return std::make_unique<InMemStrategy>(o);
}

}  // namespace kkm
