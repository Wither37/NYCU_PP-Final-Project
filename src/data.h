// Input data: dense n x d fp32 points, row-major.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kkm {

struct Dataset {
  int64_t n = 0;
  int d = 0;
  std::vector<float> X;  // n * d, row-major
  std::string name;
};

// Gaussian blobs: `centers` centers drawn from N(0, center_scale^2 I), each
// point = a center + N(0, noise^2 I). Deterministic in `seed` on any platform
// (own RNG and Box-Muller, not std::normal_distribution).
Dataset make_blobs(int64_t n, int d, int centers, uint64_t seed,
                   float center_scale = 1.0f, float noise = 0.25f);

// libSVM text format "label idx:val idx:val ...", 1-based indices, densified.
// d = largest index seen unless d_hint > 0.
Dataset load_libsvm(const std::string& path, int d_hint = 0);

// Own binary format: "KKM1", int64 n, int32 d, then n*d float32.
Dataset load_binary(const std::string& path);
void save_binary(const Dataset& ds, const std::string& path);

// Fixed initial labels, identical for every implementation:
// label[i] = splitmix64(seed + i) mod k.
std::vector<int> initial_labels(int64_t n, int k, uint64_t seed);

uint64_t splitmix64(uint64_t x);

}  // namespace kkm
