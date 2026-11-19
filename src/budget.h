// Memory budgets. Limits are never derived from observed free memory (the
// server is shared); they come from a configured budget so that runs are
// reproducible:  M_gpu = 0.9 * budget,  M_host = 0.8 * MemAvailable.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace kkm {

// Thrown when an allocation would exceed the configured budget. The driver
// reports it as "over_budget" rather than as a crash, because hitting the
// limit is an expected experimental outcome (e.g. in-memory at large n).
class BudgetError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class DeviceBudget {
 public:
  // budget_bytes is the nominal budget; 10% is held back as headroom.
  explicit DeviceBudget(size_t budget_bytes);
  ~DeviceBudget();
  DeviceBudget(const DeviceBudget&) = delete;
  DeviceBudget& operator=(const DeviceBudget&) = delete;

  void* alloc(size_t bytes, const char* what);
  template <typename T>
  T* alloc_n(size_t count, const char* what) {
    return static_cast<T*>(alloc(count * sizeof(T), what));
  }
  void free(void* p);

  size_t limit() const { return limit_; }
  size_t used() const { return used_; }
  size_t peak() const { return peak_; }

 private:
  size_t limit_, used_ = 0, peak_ = 0;
  std::unordered_map<void*, size_t> sizes_;
};

// Pinned host memory, accounted against the host budget.
class HostBudget {
 public:
  // budget_bytes == 0 means 0.8 * MemAvailable from /proc/meminfo.
  explicit HostBudget(size_t budget_bytes);
  ~HostBudget();
  HostBudget(const HostBudget&) = delete;
  HostBudget& operator=(const HostBudget&) = delete;

  void* alloc_pinned(size_t bytes, const char* what);
  void free(void* p);

  size_t limit() const { return limit_; }
  size_t used() const { return used_; }
  size_t peak() const { return peak_; }

 private:
  size_t limit_, used_ = 0, peak_ = 0;
  std::unordered_map<void*, size_t> sizes_;
};

size_t mem_available_bytes();  // MemAvailable from /proc/meminfo
std::string format_bytes(size_t bytes);

}  // namespace kkm
