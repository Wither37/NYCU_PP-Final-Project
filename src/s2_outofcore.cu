// S2: out-of-core streaming. The upper-triangular tiles of K (~2 n^2 bytes)
// live in pinned host memory and are streamed to the GPU every iteration.
//
// Pipeline: nbuf device tile buffers, one CUDA stream per buffer. Tile t uses
// buffer b = t mod nbuf and is enqueued on stream b as
//     H2D copy (tile t -> buf b)  ->  accum_rows  ->  accum_cols (off-diagonal)
// Stream order alone guarantees buf b is not overwritten before the kernels
// that read it have finished, while copies on one stream overlap kernels on
// the others. With a single H2D copy engine the copies serialise, so PCIe
// stays busy as long as the next copy is already queued; nbuf = 3 leaves
// slack for kernel jitter. Concurrent tiles hit the same rows of S, hence the
// atomic accumulation.
#include <sstream>

#include "common.h"
#include "strategy.h"
#include "tile_kernels.h"

namespace kkm {

namespace {

constexpr size_t kChunkBytes = size_t(1) << 30;  // pinned allocation granule

class S2Strategy : public SStrategy {
 public:
  explicit S2Strategy(const StrategyOptions& o) : opts_(o) {}

  ~S2Strategy() override {
    for (cudaEvent_t e : done_) cudaEventDestroy(e);
    if (begin_) cudaEventDestroy(begin_);
    for (cudaStream_t s : streams_) cudaStreamDestroy(s);
    if (dev_)
      for (float* p : dbuf_) dev_->free(p);
    if (host_)
      for (float* p : chunks_) host_->free(p);
  }

  std::string name() const override { return "s2"; }

  void setup(const StrategyContext& ctx) override {
    n_ = ctx.n;
    k_ = ctx.k;
    dev_ = ctx.dev;
    host_ = ctx.host;
    tiles_ = upper_tiles(n_, opts_.tile);

    // Device side first (small, and the likelier budget failure to report).
    const int edge = static_cast<int>(std::min<int64_t>(opts_.tile, n_));
    const size_t buf_elems = static_cast<size_t>(edge) * edge;
    const int nbuf = static_cast<int>(std::min<size_t>(opts_.nbuf, tiles_.size()));
    for (int b = 0; b < nbuf; ++b) {
      dbuf_.push_back(dev_->alloc_n<float>(buf_elems, "s2.tile_buffer"));
      cudaStream_t s;
      CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      streams_.push_back(s);
      cudaEvent_t e;
      CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      done_.push_back(e);
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&begin_, cudaEventDisableTiming));

    // Pack tiles into pinned chunks of ~1 GiB; a tile never straddles chunks.
    const size_t cap = std::max(kChunkBytes / sizeof(float), buf_elems);
    std::vector<size_t> chunk_elems;
    std::vector<std::pair<int, size_t>> where;  // (chunk, offset) per tile
    size_t cur = 0;
    for (const Tile& t : tiles_) {
      if (chunk_elems.empty() || cur + t.elems() > cap) {
        chunk_elems.push_back(0);
        cur = 0;
      }
      where.emplace_back(static_cast<int>(chunk_elems.size()) - 1, cur);
      cur += t.elems();
      chunk_elems.back() = cur;
    }
    for (size_t e : chunk_elems) {
      chunks_.push_back(static_cast<float*>(
          host_->alloc_pinned(e * sizeof(float), "s2.host_tiles")));
      pinned_bytes_ += e * sizeof(float);
    }
    for (size_t t = 0; t < tiles_.size(); ++t)
      host_ptr_.push_back(chunks_[where[t].first] + where[t].second);

    // Construction: compute each tile on the GPU, copy it down. Same ring of
    // buffers/streams, so the D2H copy of one tile overlaps building the next.
    for (size_t t = 0; t < tiles_.size(); ++t) {
      const Tile& tl = tiles_[t];
      const size_t b = t % dbuf_.size();
      launch_build_tile(ctx.d_X, ctx.d, tl.r0, tl.c0, tl.R, tl.C, dbuf_[b],
                        static_cast<size_t>(tl.C), streams_[b]);
      CUDA_CHECK(cudaMemcpyAsync(host_ptr_[t], dbuf_[b], tl.elems() * sizeof(float),
                                 cudaMemcpyDeviceToHost, streams_[b]));
    }
    for (cudaStream_t s : streams_) CUDA_CHECK(cudaStreamSynchronize(s));
  }

  void compute_S(const int* d_labels, float* d_S, cudaStream_t stream) override {
    CUDA_CHECK(cudaMemsetAsync(d_S, 0, static_cast<size_t>(n_) * k_ * sizeof(float),
                               stream));
    // Fork: internal streams start after the memset (and the caller's prior work).
    CUDA_CHECK(cudaEventRecord(begin_, stream));
    for (cudaStream_t s : streams_) CUDA_CHECK(cudaStreamWaitEvent(s, begin_, 0));

    for (size_t t = 0; t < tiles_.size(); ++t) {
      const Tile& tl = tiles_[t];
      const size_t b = t % dbuf_.size();
      cudaStream_t s = streams_[b];
      float* B = dbuf_[b];
      CUDA_CHECK(cudaMemcpyAsync(B, host_ptr_[t], tl.elems() * sizeof(float),
                                 cudaMemcpyHostToDevice, s));
      launch_accum_rows(B, tl.C, tl.R, tl.C, d_labels + tl.c0, d_S + tl.r0 * k_,
                        k_, /*atomic_out=*/true, s);
      if (tl.I != tl.J)
        launch_accum_cols(B, tl.C, tl.R, tl.C, d_labels + tl.r0,
                          d_S + tl.c0 * k_, k_, s);
    }

    // Join: the caller's stream waits for every internal stream.
    for (size_t b = 0; b < streams_.size(); ++b) {
      CUDA_CHECK(cudaEventRecord(done_[b], streams_[b]));
      CUDA_CHECK(cudaStreamWaitEvent(stream, done_[b], 0));
    }
  }

  std::string report_json() const override {
    std::ostringstream o;
    o << "\"tile\": " << opts_.tile << ", \"nbuf\": " << dbuf_.size()
      << ", \"tiles\": " << tiles_.size() << ", \"pinned_chunks\": " << chunks_.size()
      << ", \"pinned_bytes\": " << pinned_bytes_
      << ", \"h2d_bytes_per_iter\": " << pinned_bytes_;
    return o.str();
  }

 private:
  StrategyOptions opts_;
  int64_t n_ = 0;
  int k_ = 0;
  DeviceBudget* dev_ = nullptr;
  HostBudget* host_ = nullptr;
  std::vector<Tile> tiles_;
  std::vector<float*> chunks_;
  std::vector<float*> host_ptr_;
  std::vector<float*> dbuf_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> done_;
  cudaEvent_t begin_ = nullptr;
  size_t pinned_bytes_ = 0;
};

}  // namespace

std::unique_ptr<SStrategy> make_s2(const StrategyOptions& o) {
  return std::make_unique<S2Strategy>(o);
}

}  // namespace kkm
