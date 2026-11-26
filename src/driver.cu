#include "assign.h"
#include "common.h"
#include "driver.h"
#include "tile_kernels.h"

namespace kkm {

namespace {

// Device-resident inputs every strategy shares: X, diag, two label buffers, S.
struct DeviceInputs {
  DeviceBudget& dev;
  float* X = nullptr;
  float* diag = nullptr;
  int* labels[2] = {nullptr, nullptr};
  float* S = nullptr;

  DeviceInputs(DeviceBudget& b, const Dataset& ds, int k,
               const std::vector<int>& init_labels, cudaStream_t stream)
      : dev(b) {
    if (static_cast<int64_t>(init_labels.size()) != ds.n)
      throw std::invalid_argument("init_labels size != n");
    for (int v : init_labels)
      if (v < 0 || v >= k) throw std::invalid_argument("init label out of range");
    const size_t n = static_cast<size_t>(ds.n);
    X = dev.alloc_n<float>(n * ds.d, "X");
    diag = dev.alloc_n<float>(n, "diag");
    labels[0] = dev.alloc_n<int>(n, "labels0");
    labels[1] = dev.alloc_n<int>(n, "labels1");
    S = dev.alloc_n<float>(n * k, "S");
    CUDA_CHECK(cudaMemcpyAsync(X, ds.X.data(), n * ds.d * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(labels[0], init_labels.data(), n * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    launch_diag(X, ds.n, ds.d, diag, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  ~DeviceInputs() {
    dev.free(S);
    dev.free(labels[1]);
    dev.free(labels[0]);
    dev.free(diag);
    dev.free(X);
  }
};

class Stream {
 public:
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(s_); }
  operator cudaStream_t() const { return s_; }

 private:
  cudaStream_t s_;
};

}  // namespace

GpuRunResult gpu_run(const Dataset& ds, const std::vector<int>& init_labels,
                     const GpuRunOptions& opt) {
  GpuRunResult r;
  DeviceBudget dev(opt.device_budget);
  HostBudget host(opt.host_budget);
  r.device_limit = dev.limit();
  r.host_limit = host.limit();
  Stream stream;

  WallTimer tu;
  DeviceInputs in(dev, ds, opt.k, init_labels, stream);
  GpuAssign assign(ds.n, opt.k, dev);
  r.upload_ms = tu.ms();

  // Declared after the budgets, so it is destroyed (and frees) before them.
  auto strat = make_strategy(opt.strategy, opt.sopts);
  StrategyContext ctx{ds.n, ds.d, opt.k, in.X, &dev, &host};
  WallTimer ts;
  strat->setup(ctx);
  r.setup_ms = ts.ms();

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  int cur = 0;
  for (int it = 0; it < opt.iters; ++it) {
    WallTimer t;
    CUDA_CHECK(cudaEventRecord(e0, stream));
    strat->compute_S(in.labels[cur], in.S, stream);
    CUDA_CHECK(cudaEventRecord(e1, stream));
    r.iters.push_back(assign.run(in.S, in.diag, in.labels[cur], in.labels[1 - cur],
                                 stream));  // synchronises the stream
    r.iter_ms.push_back(t.ms());
    float sms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&sms, e0, e1));
    r.s_ms.push_back(sms);
    cur = 1 - cur;
  }
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));

  r.labels.resize(ds.n);
  CUDA_CHECK(cudaMemcpy(r.labels.data(), in.labels[cur], ds.n * sizeof(int),
                        cudaMemcpyDeviceToHost));
  r.strategy_json = strat->report_json();
  r.device_peak = dev.peak();
  r.host_peak = host.peak();
  return r;
}

std::vector<float> gpu_compute_S_once(const Dataset& ds, const std::vector<int>& labels,
                                      int k, const std::string& strategy,
                                      const StrategyOptions& sopts,
                                      size_t device_budget, size_t host_budget) {
  DeviceBudget dev(device_budget);
  HostBudget host(host_budget);
  Stream stream;
  DeviceInputs in(dev, ds, k, labels, stream);
  auto strat = make_strategy(strategy, sopts);
  StrategyContext ctx{ds.n, ds.d, k, in.X, &dev, &host};
  strat->setup(ctx);
  // Poison S first: a strategy that forgets to overwrite part of it shows up.
  CUDA_CHECK(cudaMemsetAsync(in.S, 0x7f, static_cast<size_t>(ds.n) * k * sizeof(float),
                             stream));
  strat->compute_S(in.labels[0], in.S, stream);
  std::vector<float> S(static_cast<size_t>(ds.n) * k);
  CUDA_CHECK(cudaMemcpyAsync(S.data(), in.S, S.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return S;
}

}  // namespace kkm
