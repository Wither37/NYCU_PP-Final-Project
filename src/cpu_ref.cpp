#include "cpu_ref.h"

#include <omp.h>

#include <limits>
#include <sstream>

#include "budget.h"
#include "common.h"

namespace kkm {

namespace {

inline double dot_d(const float* a, const float* b, int d) {
  double s = 0;
  for (int t = 0; t < d; ++t) s += static_cast<double>(a[t]) * b[t];
  return s;
}

}  // namespace

IterStats cpu_assign(const double* S, const double* diag, const int* labels_old,
                     int* labels_new, int64_t n, int k) {
  std::vector<int64_t> size(k, 0);
  std::vector<double> normsum(k, 0.0);
  for (int64_t i = 0; i < n; ++i) {
    int c = labels_old[i];
    size[c] += 1;
    normsum[c] += S[i * k + c];
  }
  std::vector<double> inv(k, 0.0), cnorm(k, 0.0);
  IterStats st;
  for (int j = 0; j < k; ++j) {
    if (size[j] == 0) continue;
    ++st.nonempty;
    inv[j] = 1.0 / static_cast<double>(size[j]);
    cnorm[j] = normsum[j] * inv[j] * inv[j];
  }
  double obj = 0;
  int64_t changed = 0;
#pragma omp parallel for reduction(+ : obj, changed) schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    double best = std::numeric_limits<double>::infinity();
    int arg = -1;
    for (int j = 0; j < k; ++j) {
      if (size[j] == 0) continue;
      double D = diag[i] - 2.0 * S[i * k + j] * inv[j] + cnorm[j];
      if (D < best) {  // strict: ties keep the smaller j
        best = D;
        arg = j;
      }
    }
    labels_new[i] = arg;
    obj += best;
    changed += (arg != labels_old[i]);
  }
  st.objective = obj;
  st.changed = changed;
  return st;
}

std::vector<double> cpu_diag(const Dataset& ds) {
  std::vector<double> diag(ds.n);
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < ds.n; ++i) {
    const float* x = &ds.X[i * ds.d];
    diag[i] = kernel_from_dot(dot_d(x, x, ds.d));
  }
  return diag;
}

void cpu_compute_S_direct(const Dataset& ds, const int* labels, int k, double* S) {
  const int64_t n = ds.n;
  const int d = ds.d;
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t i = 0; i < n; ++i) {
    double* row = S + i * k;
    for (int j = 0; j < k; ++j) row[j] = 0;
    const float* xi = &ds.X[i * d];
    for (int64_t l = 0; l < n; ++l)
      row[labels[l]] += kernel_from_dot(dot_d(xi, &ds.X[l * d], d));
  }
}

CpuKKMeans::CpuKKMeans(const Dataset& ds, int k, size_t host_budget_bytes)
    : ds_(ds), n_(ds.n), k_(k) {
  size_t need = static_cast<size_t>(n_) * n_ * sizeof(float);
  if (need > host_budget_bytes) {
    std::ostringstream m;
    m << "host budget exceeded: CPU kernel matrix needs " << format_bytes(need)
      << ", limit " << format_bytes(host_budget_bytes);
    throw BudgetError(m.str());
  }
}

void CpuKKMeans::build_K() {
  const int d = ds_.d;
  K_.resize(static_cast<size_t>(n_) * n_);
  // Upper triangle computed once, mirrored.
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t i = 0; i < n_; ++i) {
    const float* xi = &ds_.X[i * d];
    for (int64_t l = i; l < n_; ++l) {
      float v = static_cast<float>(kernel_from_dot(dot_d(xi, &ds_.X[l * d], d)));
      K_[i * n_ + l] = v;
      K_[l * n_ + i] = v;
    }
  }
  diag_ = cpu_diag(ds_);
}

void CpuKKMeans::compute_S(const int* labels, double* S) const {
  const int k = k_;
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n_; ++i) {
    double* row = S + i * k;
    for (int j = 0; j < k; ++j) row[j] = 0;
    const float* Ki = &K_[i * n_];
    for (int64_t l = 0; l < n_; ++l) row[labels[l]] += Ki[l];
  }
}

CpuRunResult cpu_run(const Dataset& ds, int k, int iters,
                     const std::vector<int>& init_labels,
                     size_t host_budget_bytes) {
  CpuRunResult r;
  CpuKKMeans cpu(ds, k, host_budget_bytes);
  WallTimer tb;
  cpu.build_K();
  r.build_ms = tb.ms();

  std::vector<int> cur = init_labels, next(ds.n);
  std::vector<double> S(static_cast<size_t>(ds.n) * k);
  for (int it = 0; it < iters; ++it) {
    WallTimer t;
    cpu.compute_S(cur.data(), S.data());
    r.iters.push_back(cpu_assign(S.data(), cpu.diag().data(), cur.data(),
                                 next.data(), ds.n, k));
    r.iter_ms.push_back(t.ms());
    cur.swap(next);
  }
  r.labels = std::move(cur);
  return r;
}

}  // namespace kkm
