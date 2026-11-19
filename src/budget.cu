#include "budget.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include "common.h"

namespace kkm {

std::string format_bytes(size_t bytes) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.3f GiB", bytes / kGiB);
  return buf;
}

size_t mem_available_bytes() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  size_t kb = 0;
  std::string unit;
  while (f >> key >> kb >> unit) {
    if (key == "MemAvailable:") return kb * 1024;
  }
  throw std::runtime_error("MemAvailable not found in /proc/meminfo");
}

// ---------------------------------------------------------------- device ---

DeviceBudget::DeviceBudget(size_t budget_bytes)
    : limit_(static_cast<size_t>(0.9 * static_cast<double>(budget_bytes))) {}

DeviceBudget::~DeviceBudget() {
  for (auto& [p, n] : sizes_) cudaFree(p);
}

void* DeviceBudget::alloc(size_t bytes, const char* what) {
  if (bytes == 0) bytes = 1;
  if (used_ + bytes > limit_) {
    std::ostringstream m;
    m << "device budget exceeded allocating " << what << " ("
      << format_bytes(bytes) << "): in use " << format_bytes(used_)
      << ", limit " << format_bytes(limit_);
    throw BudgetError(m.str());
  }
  void* p = nullptr;
  cudaError_t err = cudaMalloc(&p, bytes);
  if (err != cudaSuccess) {
    cudaGetLastError();  // clear the sticky-free error state
    std::ostringstream m;
    m << "cudaMalloc failed for " << what << " (" << format_bytes(bytes)
      << ") inside the budget: " << cudaGetErrorString(err)
      << " -- another process is probably using this GPU";
    throw std::runtime_error(m.str());
  }
  sizes_[p] = bytes;
  used_ += bytes;
  if (used_ > peak_) peak_ = used_;
  return p;
}

void DeviceBudget::free(void* p) {
  if (!p) return;
  auto it = sizes_.find(p);
  if (it == sizes_.end()) throw std::logic_error("DeviceBudget::free: unknown pointer");
  CUDA_CHECK(cudaFree(p));
  used_ -= it->second;
  sizes_.erase(it);
}

// ------------------------------------------------------------------ host ---

HostBudget::HostBudget(size_t budget_bytes)
    : limit_(budget_bytes ? budget_bytes
                          : static_cast<size_t>(0.8 * mem_available_bytes())) {}

HostBudget::~HostBudget() {
  for (auto& [p, n] : sizes_) cudaFreeHost(p);
}

void* HostBudget::alloc_pinned(size_t bytes, const char* what) {
  if (bytes == 0) bytes = 1;
  if (used_ + bytes > limit_) {
    std::ostringstream m;
    m << "host budget exceeded allocating " << what << " ("
      << format_bytes(bytes) << "): pinned " << format_bytes(used_)
      << ", limit " << format_bytes(limit_);
    throw BudgetError(m.str());
  }
  void* p = nullptr;
  cudaError_t err = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaGetLastError();
    std::ostringstream m;
    m << "cudaHostAlloc failed for " << what << " (" << format_bytes(bytes)
      << ") after pinning " << format_bytes(used_) << ": "
      << cudaGetErrorString(err);
    throw std::runtime_error(m.str());
  }
  sizes_[p] = bytes;
  used_ += bytes;
  if (used_ > peak_) peak_ = used_;
  return p;
}

void HostBudget::free(void* p) {
  if (!p) return;
  auto it = sizes_.find(p);
  if (it == sizes_.end()) throw std::logic_error("HostBudget::free: unknown pointer");
  CUDA_CHECK(cudaFreeHost(p));
  used_ -= it->second;
  sizes_.erase(it);
}

}  // namespace kkm
