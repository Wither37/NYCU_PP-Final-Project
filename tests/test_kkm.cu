// Correctness tests. GPU strategies are checked against the CPU reference on
// small inputs that exercise uneven tiles, diagonal tiles, k = 1, k too large
// for the shared accumulator, empty clusters and ties.
//
//   ./bin/test_kkm          all tests (needs a GPU)
//   ./bin/test_kkm --cpu    CPU-only tests
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "assign.h"
#include "budget.h"
#include "common.h"
#include "cpu_ref.h"
#include "data.h"
#include "driver.h"
#include "strategy.h"

using namespace kkm;

namespace {

int g_failed = 0, g_checks = 0;

#define EXPECT(cond, ...)                                       \
  do {                                                          \
    ++g_checks;                                                 \
    if (!(cond)) {                                              \
      ++g_failed;                                               \
      std::printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__);                                 \
      std::printf("\n");                                        \
    }                                                           \
  } while (0)

constexpr size_t kBigBudget = size_t(8) << 30;
constexpr size_t kHostBudget = size_t(4) << 30;

// Max error of S against the reference, relative to each row's total
// (all K entries are >= 0 for this kernel, so the row total bounds |S[i][j]|).
double s_error(const std::vector<float>& S, const std::vector<double>& ref,
               int64_t n, int k) {
  double worst = 0;
  for (int64_t i = 0; i < n; ++i) {
    double tot = 0;
    for (int j = 0; j < k; ++j) tot += std::fabs(ref[i * k + j]);
    for (int j = 0; j < k; ++j) {
      double e = std::fabs(S[i * k + j] - ref[i * k + j]) / std::max(tot, 1e-30);
      if (!(e <= worst)) worst = e;  // also catches NaN (the poison value)
    }
  }
  return worst;
}

void test_tiles() {
  std::puts("tiles: upper-triangular tiling covers each block pair once");
  for (int64_t n : {1, 63, 64, 1000, 4097}) {
    for (int T : {64, 256, 4096}) {
      auto tiles = upper_tiles(n, T);
      size_t elems = 0, diag = 0;
      for (auto& t : tiles) {
        EXPECT(t.I <= t.J, "tile (%d,%d)", t.I, t.J);
        EXPECT(t.R > 0 && t.C > 0 && t.r0 + t.R <= n && t.c0 + t.C <= n,
               "n=%lld T=%d bad bounds", (long long)n, T);
        elems += t.elems();
        if (t.I == t.J) diag += t.elems();
      }
      // Off-diagonal tiles count twice, diagonal once: total must be n^2.
      EXPECT(2 * (elems - diag) + diag == static_cast<size_t>(n * n),
             "n=%lld T=%d coverage", (long long)n, T);
    }
  }
}

void test_cpu_reference() {
  std::puts("cpu: stored-K S equals on-the-fly S");
  Dataset ds = make_blobs(300, 7, 4, 11);
  const int k = 5;
  auto labels = initial_labels(ds.n, k, 3);
  CpuKKMeans cpu(ds, k, kHostBudget);
  cpu.build_K();
  std::vector<double> a(ds.n * k), b(ds.n * k);
  cpu.compute_S(labels.data(), a.data());
  cpu_compute_S_direct(ds, labels.data(), k, b.data());
  std::vector<float> af(a.begin(), a.end());
  double e = s_error(af, b, ds.n, k);
  EXPECT(e < 1e-6, "rel err %.3g", e);
}

// Hand-made S where clusters 1 and 2 tie exactly and cluster 0 is empty but
// would win if it were not excluded.
struct TieCase {
  int64_t n = 4;
  int k = 4;
  std::vector<int> old_labels{1, 2, 1, 2};
  std::vector<double> S, diag{1, 1, 1, 1};
  TieCase() {
    S.assign(n * k, 0.0);
    for (int64_t i = 0; i < n; ++i) {
      S[i * k + 0] = 100;  // empty cluster, huge similarity
      S[i * k + 1] = 5;
      S[i * k + 2] = 5;
      S[i * k + 3] = 0;    // empty
    }
  }
};

void test_cpu_assign_rules() {
  std::puts("cpu: ties go to the smallest j, empty clusters are excluded");
  TieCase tc;
  std::vector<int> out(tc.n);
  IterStats st = cpu_assign(tc.S.data(), tc.diag.data(), tc.old_labels.data(),
                            out.data(), tc.n, tc.k);
  for (int64_t i = 0; i < tc.n; ++i) EXPECT(out[i] == 1, "point %lld -> %d", (long long)i, out[i]);
  EXPECT(st.nonempty == 2, "nonempty %d", st.nonempty);
  EXPECT(st.changed == 2, "changed %lld", (long long)st.changed);
}

void test_gpu_assign() {
  std::puts("gpu: assignment module matches cpu_assign");
  // Same tie case on the GPU.
  {
    TieCase tc;
    DeviceBudget dev(kBigBudget);
    GpuAssign ga(tc.n, tc.k, dev);
    std::vector<float> Sf(tc.S.begin(), tc.S.end()), df(tc.diag.begin(), tc.diag.end());
    float* S = dev.alloc_n<float>(Sf.size(), "S");
    float* diag = dev.alloc_n<float>(tc.n, "diag");
    int* lo = dev.alloc_n<int>(tc.n, "lo");
    int* ln = dev.alloc_n<int>(tc.n, "ln");
    CUDA_CHECK(cudaMemcpy(S, Sf.data(), Sf.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(diag, df.data(), tc.n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lo, tc.old_labels.data(), tc.n * 4, cudaMemcpyHostToDevice));
    IterStats st = ga.run(S, diag, lo, ln, 0);
    std::vector<int> out(tc.n);
    CUDA_CHECK(cudaMemcpy(out.data(), ln, tc.n * 4, cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < tc.n; ++i) EXPECT(out[i] == 1, "point %lld -> %d", (long long)i, out[i]);
    EXPECT(st.nonempty == 2, "nonempty %d", st.nonempty);
    EXPECT(st.changed == 2, "changed %lld", (long long)st.changed);
    for (void* p : {(void*)S, (void*)diag, (void*)lo, (void*)ln}) dev.free(p);
  }
  // Random case: identical fp32 inputs to both sides, labels must agree.
  {
    Dataset ds = make_blobs(5000, 6, 8, 21);
    const int k = 8;
    auto labels = initial_labels(ds.n, k, 5);
    std::vector<double> Sd(ds.n * k);
    cpu_compute_S_direct(ds, labels.data(), k, Sd.data());
    std::vector<float> Sf(Sd.begin(), Sd.end());
    std::vector<double> Sr(Sf.begin(), Sf.end());  // what the GPU sees
    std::vector<double> diag = cpu_diag(ds);
    std::vector<float> df(diag.begin(), diag.end());
    std::vector<double> dr(df.begin(), df.end());
    std::vector<int> cpu_out(ds.n);
    IterStats cs = cpu_assign(Sr.data(), dr.data(), labels.data(), cpu_out.data(), ds.n, k);

    DeviceBudget dev(kBigBudget);
    GpuAssign ga(ds.n, k, dev);
    float* S = dev.alloc_n<float>(Sf.size(), "S");
    float* dg = dev.alloc_n<float>(ds.n, "diag");
    int* lo = dev.alloc_n<int>(ds.n, "lo");
    int* ln = dev.alloc_n<int>(ds.n, "ln");
    CUDA_CHECK(cudaMemcpy(S, Sf.data(), Sf.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, df.data(), ds.n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lo, labels.data(), ds.n * 4, cudaMemcpyHostToDevice));
    IterStats gs = ga.run(S, dg, lo, ln, 0);
    std::vector<int> gpu_out(ds.n);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), ln, ds.n * 4, cudaMemcpyDeviceToHost));
    int64_t diff = 0;
    for (int64_t i = 0; i < ds.n; ++i) diff += (gpu_out[i] != cpu_out[i]);
    EXPECT(diff == 0, "%lld labels differ", (long long)diff);
    EXPECT(gs.changed == cs.changed, "changed %lld vs %lld", (long long)gs.changed,
           (long long)cs.changed);
    double rel = std::fabs(gs.objective - cs.objective) / std::fabs(cs.objective);
    EXPECT(rel < 1e-9, "objective rel diff %.3g", rel);
    for (void* p : {(void*)S, (void*)dg, (void*)lo, (void*)ln}) dev.free(p);
  }
}

struct SCase {
  int64_t n;
  int d, k, tile, nbuf;
  const char* what;
  std::vector<int> only;  // if nonempty, labels restricted to these clusters
};

void test_gpu_S() {
  std::puts("gpu: S from every strategy matches the CPU reference");
  const std::vector<SCase> cases = {
      {1000, 8, 7, 256, 3, "uneven last tile (232)", {}},
      {513, 3, 1, 128, 3, "k = 1", {}},
      {777, 50, 10, 200, 3, "d = 50, tile not a multiple of 64", {}},
      {300, 16, 20, 512, 3, "single tile larger than n", {}},
      {1100, 5, 400, 256, 3, "k = 400: accum_cols global fallback", {}},
      {640, 4, 12, 128, 3, "empty clusters", {0, 3, 5, 11}},
      {1000, 8, 7, 128, 1, "s2 with one buffer", {}},
      {1000, 8, 7, 128, 2, "s2 with two buffers", {}},
  };
  for (const auto& c : cases) {
    Dataset ds = make_blobs(c.n, c.d, 5, 100 + c.n);
    std::vector<int> labels = initial_labels(ds.n, c.k, 7);
    if (!c.only.empty())
      for (auto& v : labels) v = c.only[v % c.only.size()];
    std::vector<double> ref(ds.n * c.k);
    cpu_compute_S_direct(ds, labels.data(), c.k, ref.data());
    for (const char* strat : {"inmem", "s1", "s2"}) {
      StrategyOptions so;
      so.tile = c.tile;
      so.nbuf = c.nbuf;
      auto S = gpu_compute_S_once(ds, labels, c.k, strat, so, kBigBudget, kHostBudget);
      double e = s_error(S, ref, ds.n, c.k);
      std::printf("  %-6s n=%-5lld d=%-3d k=%-4d T=%-4d nbuf=%d  err %.2e  (%s)\n", strat,
                  (long long)c.n, c.d, c.k, c.tile, c.nbuf, e, c.what);
      EXPECT(e < 5e-6, "%s: %s rel err %.3g", strat, c.what, e);
    }
  }
}

void test_gpu_full_run() {
  std::puts("gpu: 30-iteration runs agree with the CPU reference");
  Dataset ds = make_blobs(3000, 10, 10, 42, 1.0f, 0.4f);
  const int k = 10, iters = 30;
  auto init = initial_labels(ds.n, k, 9);
  CpuRunResult ref = cpu_run(ds, k, iters, init, kHostBudget);
  for (const char* strat : {"inmem", "s1", "s2"}) {
    GpuRunOptions o;
    o.strategy = strat;
    o.k = k;
    o.iters = iters;
    o.device_budget = kBigBudget;
    o.host_budget = kHostBudget;
    o.sopts.tile = 512;
    GpuRunResult r = gpu_run(ds, init, o);
    int64_t agree = 0;
    for (int64_t i = 0; i < ds.n; ++i) agree += (r.labels[i] == ref.labels[i]);
    double a = double(agree) / ds.n;
    double rel = std::fabs(r.iters.back().objective - ref.iters.back().objective) /
                 std::fabs(ref.iters.back().objective);
    int first_changed_diff = static_cast<int>(r.iters[0].changed - ref.iters[0].changed);
    std::printf("  %-6s agreement %.5f  objective rel diff %.2e  iter-1 changed %lld vs %lld\n",
                strat, a, rel, (long long)r.iters[0].changed, (long long)ref.iters[0].changed);
    EXPECT(a >= 0.999, "%s agreement %.5f", strat, a);
    EXPECT(rel < 1e-5, "%s objective rel %.3g", strat, rel);
    EXPECT(std::abs(first_changed_diff) <= 1, "%s first-iteration changes differ", strat);
  }
}

void test_budget() {
  std::puts("budget: over-budget configurations are rejected, not crashed");
  Dataset ds = make_blobs(4000, 4, 4, 1);
  auto init = initial_labels(ds.n, 4, 1);
  GpuRunOptions o;
  o.k = 4;
  o.iters = 1;
  o.host_budget = kHostBudget;
  // inmem needs 4 * 4000^2 B = 61 MiB for K; give it 40 MiB nominal.
  o.strategy = "inmem";
  o.device_budget = size_t(40) << 20;
  bool threw = false;
  try { gpu_run(ds, init, o); } catch (const BudgetError&) { threw = true; }
  EXPECT(threw, "inmem should exceed a 40 MiB budget");
  // s2 at the same device budget fits (tile buffers only) ...
  o.strategy = "s2";
  o.sopts.tile = 1024;
  threw = false;
  try { gpu_run(ds, init, o); } catch (const BudgetError&) { threw = true; }
  EXPECT(!threw, "s2 should fit a 40 MiB device budget");
  // ... but not a 16 MiB host budget (needs ~ 2 * 61 MiB / 2 + diagonal).
  o.host_budget = size_t(16) << 20;
  threw = false;
  try { gpu_run(ds, init, o); } catch (const BudgetError&) { threw = true; }
  EXPECT(threw, "s2 should exceed a 16 MiB host budget");
}

}  // namespace

int main(int argc, char** argv) {
  bool cpu_only = argc > 1 && std::strcmp(argv[1], "--cpu") == 0;
  test_tiles();
  test_cpu_reference();
  test_cpu_assign_rules();
  if (!cpu_only) {
    test_gpu_assign();
    test_gpu_S();
    test_gpu_full_run();
    test_budget();
  }
  std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
  return g_failed ? 1 : 0;
}
