// S1: only the upper-triangular tiles of K are resident on the GPU (~2 n^2
// bytes). One pass over an off-diagonal tile produces both its row
// contribution (rows of I) and its mirrored column contribution (rows of J);
// both land in S through atomics. Stretch goal in the proposal.
#include <sstream>

#include "common.h"
#include "strategy.h"
#include "tile_kernels.h"

namespace kkm {

namespace {

class S1Strategy : public SStrategy {
 public:
  explicit S1Strategy(const StrategyOptions& o) : opts_(o) {}
  ~S1Strategy() override {
    if (dev_)
      for (float* p : tile_ptr_) dev_->free(p);
  }

  std::string name() const override { return "s1"; }

  void setup(const StrategyContext& ctx) override {
    n_ = ctx.n;
    k_ = ctx.k;
    dev_ = ctx.dev;
    tiles_ = upper_tiles(n_, opts_.tile);
    tile_ptr_.reserve(tiles_.size());
    for (const Tile& t : tiles_) {
      float* p = dev_->alloc_n<float>(t.elems(), "s1.tile");
      tile_ptr_.push_back(p);
      bytes_ += t.elems() * sizeof(float);
      launch_build_tile(ctx.d_X, ctx.d, t.r0, t.c0, t.R, t.C, p,
                        static_cast<size_t>(t.C), 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void compute_S(const int* d_labels, float* d_S, cudaStream_t stream) override {
    CUDA_CHECK(cudaMemsetAsync(d_S, 0, static_cast<size_t>(n_) * k_ * sizeof(float),
                               stream));
    for (size_t t = 0; t < tiles_.size(); ++t) {
      const Tile& tl = tiles_[t];
      const float* B = tile_ptr_[t];
      launch_accum_rows(B, tl.C, tl.R, tl.C, d_labels + tl.c0, d_S + tl.r0 * k_,
                        k_, /*atomic_out=*/true, stream);
      if (tl.I != tl.J)
        launch_accum_cols(B, tl.C, tl.R, tl.C, d_labels + tl.r0,
                          d_S + tl.c0 * k_, k_, stream);
    }
  }

  std::string report_json() const override {
    std::ostringstream o;
    o << "\"tile\": " << opts_.tile << ", \"tiles\": " << tiles_.size()
      << ", \"K_device_bytes\": " << bytes_;
    return o.str();
  }

 private:
  StrategyOptions opts_;
  int64_t n_ = 0;
  int k_ = 0;
  DeviceBudget* dev_ = nullptr;
  std::vector<Tile> tiles_;
  std::vector<float*> tile_ptr_;
  size_t bytes_ = 0;
};

}  // namespace

std::unique_ptr<SStrategy> make_s1(const StrategyOptions& o) {
  return std::make_unique<S1Strategy>(o);
}

}  // namespace kkm
