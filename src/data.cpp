#include "data.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace kkm {

uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

namespace {

class Rng {
 public:
  explicit Rng(uint64_t seed) : s_(seed) {}
  uint64_t next() { return splitmix64(s_++); }
  // Uniform in (0, 1].
  double uniform() { return ((next() >> 11) + 1) * (1.0 / 9007199254740992.0); }
  double normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    double u1 = uniform(), u2 = uniform();
    double r = std::sqrt(-2.0 * std::log(u1));
    double t = 2.0 * M_PI * u2;
    spare_ = r * std::sin(t);
    has_spare_ = true;
    return r * std::cos(t);
  }

 private:
  uint64_t s_;
  bool has_spare_ = false;
  double spare_ = 0;
};

}  // namespace

Dataset make_blobs(int64_t n, int d, int centers, uint64_t seed,
                   float center_scale, float noise) {
  if (n <= 0 || d <= 0 || centers <= 0)
    throw std::invalid_argument("make_blobs: n, d, centers must be positive");
  Rng rng(seed);
  std::vector<double> C(static_cast<size_t>(centers) * d);
  for (auto& c : C) c = center_scale * rng.normal();
  Dataset ds;
  ds.n = n;
  ds.d = d;
  ds.X.resize(static_cast<size_t>(n) * d);
  for (int64_t i = 0; i < n; ++i) {
    int c = static_cast<int>(rng.next() % centers);
    for (int t = 0; t < d; ++t)
      ds.X[i * d + t] = static_cast<float>(C[c * d + t] + noise * rng.normal());
  }
  char name[96];
  std::snprintf(name, sizeof name, "blobs(n=%lld,d=%d,c=%d,seed=%llu)",
                static_cast<long long>(n), d, centers,
                static_cast<unsigned long long>(seed));
  ds.name = name;
  return ds;
}

Dataset load_libsvm(const std::string& path, int d_hint) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<std::vector<std::pair<int, float>>> rows;
  int dmax = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string tok;
    ss >> tok;  // class label, unused
    std::vector<std::pair<int, float>> r;
    while (ss >> tok) {
      auto colon = tok.find(':');
      if (colon == std::string::npos) continue;
      int idx = std::stoi(tok.substr(0, colon));
      float v = std::stof(tok.substr(colon + 1));
      if (idx < 1) throw std::runtime_error("libsvm index < 1 in " + path);
      r.emplace_back(idx - 1, v);
      if (idx > dmax) dmax = idx;
    }
    rows.push_back(std::move(r));
  }
  Dataset ds;
  ds.n = static_cast<int64_t>(rows.size());
  ds.d = d_hint > 0 ? d_hint : dmax;
  if (ds.n == 0 || ds.d == 0) throw std::runtime_error("empty dataset " + path);
  ds.X.assign(static_cast<size_t>(ds.n) * ds.d, 0.0f);
  for (int64_t i = 0; i < ds.n; ++i)
    for (auto [j, v] : rows[i])
      if (j < ds.d) ds.X[i * ds.d + j] = v;
  ds.name = path;
  return ds;
}

Dataset load_binary(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "KKM1", 4) != 0)
    throw std::runtime_error(path + ": not a KKM1 file");
  Dataset ds;
  int32_t d32 = 0;
  f.read(reinterpret_cast<char*>(&ds.n), sizeof ds.n);
  f.read(reinterpret_cast<char*>(&d32), sizeof d32);
  ds.d = d32;
  if (!f || ds.n <= 0 || ds.d <= 0) throw std::runtime_error(path + ": bad header");
  ds.X.resize(static_cast<size_t>(ds.n) * ds.d);
  f.read(reinterpret_cast<char*>(ds.X.data()), ds.X.size() * sizeof(float));
  if (!f) throw std::runtime_error(path + ": truncated");
  ds.name = path;
  return ds;
}

void save_binary(const Dataset& ds, const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  int32_t d32 = ds.d;
  f.write("KKM1", 4);
  f.write(reinterpret_cast<const char*>(&ds.n), sizeof ds.n);
  f.write(reinterpret_cast<const char*>(&d32), sizeof d32);
  f.write(reinterpret_cast<const char*>(ds.X.data()), ds.X.size() * sizeof(float));
  if (!f) throw std::runtime_error("write failed: " + path);
}

std::vector<int> initial_labels(int64_t n, int k, uint64_t seed) {
  std::vector<int> labels(n);
  for (int64_t i = 0; i < n; ++i)
    labels[i] = static_cast<int>(splitmix64(seed + static_cast<uint64_t>(i)) % k);
  return labels;
}

}  // namespace kkm
