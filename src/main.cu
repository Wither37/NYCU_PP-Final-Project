// kkm -- exact Kernel K-means with a selectable strategy for computing S.
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

#include "budget.h"
#include "common.h"
#include "cpu_ref.h"
#include "data.h"
#include "driver.h"

using namespace kkm;

namespace {

void usage() {
  std::cerr <<
      R"(usage: kkm [options]
  input (one of):
    --synthetic N D       Gaussian blobs, N points in D dims
      --centers C         number of blob centers (default: k)
    --libsvm PATH         libSVM text file (densified)
    --input PATH          KKM1 binary file (see --save-data)
  algorithm:
    --strategy NAME       cpu | inmem | s1 | s2          (default s2)
    --k K                 clusters, 1..1024              (default 10)
    --iters I             fixed iteration count          (default 30)
    --seed S              data and initial-label seed    (default 1)
  memory:
    --budget-gb G         nominal GPU budget; 90% usable (default 24)
    --host-budget-gb G    host budget (default 0.8 * MemAvailable)
  strategy tuning:
    --tile T              tile edge for s1/s2            (default 4096)
    --nbuf B              s2 device buffers/streams      (default 3)
  misc:
    --threads T           OpenMP threads for cpu/--check (default: all)
    --check               also run the CPU reference and compare
    --json PATH           write the run report as JSON
    --save-data PATH      write the input as KKM1 binary and exit
)";
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

double mean(const std::vector<double>& v) {
  return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Round-trip precision: the default 6 significant digits would make objectives
// from different implementations look identical when they are not.
constexpr int kJsonDigits = std::numeric_limits<double>::max_digits10;

// JSON has no NaN/Inf; write null so the report stays parseable.
std::string json_num(double v) {
  if (!std::isfinite(v)) return "null";
  std::ostringstream o;
  o << std::setprecision(kJsonDigits) << v;
  return o.str();
}

std::string json_array(const std::vector<double>& v) {
  std::ostringstream o;
  o << "[";
  for (size_t i = 0; i < v.size(); ++i) o << (i ? ", " : "") << json_num(v[i]);
  return o.str() + "]";
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  std::string strategy = "s2", libsvm, input, json_path, save_data;
  int64_t syn_n = 0;
  int syn_d = 0, centers = 0, k = 10, iters = 30, threads = 0;
  uint64_t seed = 1;
  double budget_gb = 24, host_budget_gb = 0;
  bool check = false;
  StrategyOptions sopts;

  for (int a = 1; a < argc; ++a) {
    std::string o = argv[a];
    auto need = [&](int cnt) {
      if (a + cnt >= argc) {
        std::cerr << "missing value for " << o << "\n";
        std::exit(1);
      }
    };
    if (o == "--synthetic") { need(2); syn_n = std::stoll(argv[++a]); syn_d = std::stoi(argv[++a]); }
    else if (o == "--centers") { need(1); centers = std::stoi(argv[++a]); }
    else if (o == "--libsvm") { need(1); libsvm = argv[++a]; }
    else if (o == "--input") { need(1); input = argv[++a]; }
    else if (o == "--strategy") { need(1); strategy = argv[++a]; }
    else if (o == "--k") { need(1); k = std::stoi(argv[++a]); }
    else if (o == "--iters") { need(1); iters = std::stoi(argv[++a]); }
    else if (o == "--seed") { need(1); seed = std::stoull(argv[++a]); }
    else if (o == "--budget-gb") { need(1); budget_gb = std::stod(argv[++a]); }
    else if (o == "--host-budget-gb") { need(1); host_budget_gb = std::stod(argv[++a]); }
    else if (o == "--tile") { need(1); sopts.tile = std::stoi(argv[++a]); }
    else if (o == "--nbuf") { need(1); sopts.nbuf = std::stoi(argv[++a]); }
    else if (o == "--threads") { need(1); threads = std::stoi(argv[++a]); }
    else if (o == "--check") { check = true; }
    else if (o == "--json") { need(1); json_path = argv[++a]; }
    else if (o == "--save-data") { need(1); save_data = argv[++a]; }
    else if (o == "-h" || o == "--help") { usage(); return 0; }
    else { std::cerr << "unknown option " << o << "\n"; usage(); return 1; }
  }
  if (k < 1 || k > kMaxK) { std::cerr << "--k must be in [1, 1024]\n"; return 1; }
  if (threads > 0) omp_set_num_threads(threads);

  Dataset ds;
  try {
    if (syn_n > 0) ds = make_blobs(syn_n, syn_d, centers > 0 ? centers : k, seed);
    else if (!libsvm.empty()) ds = load_libsvm(libsvm);
    else if (!input.empty()) ds = load_binary(input);
    else { usage(); return 1; }
  } catch (const std::exception& e) {
    std::cerr << "input error: " << e.what() << "\n";
    return 1;
  }
  if (!save_data.empty()) {
    save_binary(ds, save_data);
    std::cout << "wrote " << save_data << " (n=" << ds.n << ", d=" << ds.d << ")\n";
    return 0;
  }

  // Labels use a seed stream separate from the data generator's.
  const std::vector<int> init = initial_labels(ds.n, k, seed ^ 0x5eedull);
  const size_t dev_budget = static_cast<size_t>(budget_gb * kGiB);
  const size_t host_budget = host_budget_gb > 0
                                 ? static_cast<size_t>(host_budget_gb * kGiB)
                                 : static_cast<size_t>(0.8 * mem_available_bytes());

  std::cout << "dataset " << ds.name << "  n=" << ds.n << " d=" << ds.d
            << " k=" << k << " iters=" << iters << " strategy=" << strategy << "\n";

  std::ostringstream js;
  js << std::setprecision(kJsonDigits);
  js << "{\n  \"dataset\": \"" << json_escape(ds.name) << "\", \"n\": " << ds.n
     << ", \"d\": " << ds.d << ", \"k\": " << k << ", \"iters\": " << iters
     << ", \"seed\": " << seed << ",\n  \"strategy\": \"" << strategy << "\""
     // Nominal budgets are recorded even when the run ends over_budget.
     << ", \"budget_bytes\": " << dev_budget << ", \"host_budget_bytes\": " << host_budget;

  std::vector<int> labels;
  std::vector<IterStats> its;
  int status = 0;
  try {
    if (strategy == "cpu") {
      WallTimer tt;
      CpuRunResult r = cpu_run(ds, k, iters, init, host_budget);
      double total = tt.ms();
      labels = r.labels;
      its = r.iters;
      std::printf("build %.2f ms | iter median %.3f ms mean %.3f ms | total %.2f ms | threads %d\n",
                  r.build_ms, median(r.iter_ms), mean(r.iter_ms), total,
                  omp_get_max_threads());
      js << ", \"status\": \"ok\", \"threads\": " << omp_get_max_threads()
         << ",\n  \"build_ms\": " << r.build_ms << ", \"total_ms\": " << total
         << ",\n  \"iter_ms\": " << json_array(r.iter_ms);
    } else {
      GpuRunOptions opt;
      opt.strategy = strategy;
      opt.k = k;
      opt.iters = iters;
      opt.device_budget = dev_budget;
      opt.host_budget = host_budget;
      opt.sopts = sopts;
      int dev = 0;
      CUDA_CHECK(cudaGetDevice(&dev));
      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
      WallTimer tt;
      GpuRunResult r = gpu_run(ds, init, opt);
      double total = tt.ms();
      labels = r.labels;
      its = r.iters;
      std::printf("gpu %s | upload %.2f ms | setup %.2f ms | S median %.3f ms | "
                  "iter median %.3f ms mean %.3f ms | total %.2f ms\n",
                  prop.name, r.upload_ms, r.setup_ms, median(r.s_ms),
                  median(r.iter_ms), mean(r.iter_ms), total);
      std::printf("device peak %s of %s | pinned peak %s of %s\n",
                  format_bytes(r.device_peak).c_str(), format_bytes(r.device_limit).c_str(),
                  format_bytes(r.host_peak).c_str(), format_bytes(r.host_limit).c_str());
      js << ", \"status\": \"ok\", \"gpu\": \"" << json_escape(prop.name) << "\""
         << ",\n  \"device_limit\": " << r.device_limit << ", \"device_peak\": " << r.device_peak
         << ", \"host_limit\": " << r.host_limit << ", \"host_peak\": " << r.host_peak
         << ",\n  \"upload_ms\": " << r.upload_ms << ", \"setup_ms\": " << r.setup_ms
         << ", \"total_ms\": " << total
         << ",\n  \"s_ms\": " << json_array(r.s_ms)
         << ",\n  \"iter_ms\": " << json_array(r.iter_ms);
      if (!r.strategy_json.empty()) js << ",\n  " << r.strategy_json;
    }
  } catch (const BudgetError& e) {
    std::cout << "OVER BUDGET: " << e.what() << "\n";
    js << ", \"status\": \"over_budget\", \"error\": \"" << json_escape(e.what()) << "\"";
    status = 3;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    js << ", \"status\": \"error\", \"error\": \"" << json_escape(e.what()) << "\"";
    status = 2;
  }

  if (status == 0) {
    std::vector<double> obj;
    for (auto& s : its) obj.push_back(s.objective);
    js << ",\n  \"objective\": " << json_array(obj);
    if (!its.empty())
      std::printf("objective first %.9g last %.9g | changed in last iter %lld | nonempty %d\n",
                  its.front().objective, its.back().objective,
                  static_cast<long long>(its.back().changed), its.back().nonempty);
  }

  if (status == 0 && check && strategy != "cpu") {
    CpuRunResult ref = cpu_run(ds, k, iters, init, host_budget);
    int64_t agree = 0;
    for (int64_t i = 0; i < ds.n; ++i) agree += (ref.labels[i] == labels[i]);
    double a = static_cast<double>(agree) / ds.n;
    double o_gpu = its.back().objective, o_cpu = ref.iters.back().objective;
    double rel = std::fabs(o_gpu - o_cpu) / std::max(1e-30, std::fabs(o_cpu));
    std::printf("check vs cpu: label agreement %.6f | objective rel diff %.3g\n", a, rel);
    js << ",\n  \"check_label_agreement\": " << json_num(a)
       << ", \"check_objective_rel\": " << json_num(rel);
  }
  js << "\n}\n";

  if (!json_path.empty()) {
    std::ofstream f(json_path);
    f << js.str();
  }
  return status;
}
