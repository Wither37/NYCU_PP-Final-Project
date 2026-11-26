#include <stdexcept>

#include "common.h"
#include "strategy.h"

namespace kkm {

std::unique_ptr<SStrategy> make_inmem(const StrategyOptions&);
std::unique_ptr<SStrategy> make_s1(const StrategyOptions&);
std::unique_ptr<SStrategy> make_s2(const StrategyOptions&);

std::unique_ptr<SStrategy> make_strategy(const std::string& name,
                                         const StrategyOptions& opts) {
  if (opts.tile < 64) throw std::invalid_argument("tile must be >= 64");
  if (opts.nbuf < 1) throw std::invalid_argument("nbuf must be >= 1");
  if (name == "inmem") return make_inmem(opts);
  if (name == "s1") return make_s1(opts);
  if (name == "s2") return make_s2(opts);
  throw std::invalid_argument("unknown strategy '" + name + "'");
}

std::vector<Tile> upper_tiles(int64_t n, int T) {
  const int nt = static_cast<int>(ceil_div(n, T));
  std::vector<Tile> tiles;
  tiles.reserve(static_cast<size_t>(nt) * (nt + 1) / 2);
  for (int I = 0; I < nt; ++I) {
    for (int J = I; J < nt; ++J) {
      Tile t;
      t.I = I;
      t.J = J;
      t.r0 = static_cast<int64_t>(I) * T;
      t.c0 = static_cast<int64_t>(J) * T;
      t.R = static_cast<int>(std::min<int64_t>(T, n - t.r0));
      t.C = static_cast<int>(std::min<int64_t>(T, n - t.c0));
      tiles.push_back(t);
    }
  }
  return tiles;
}

}  // namespace kkm
