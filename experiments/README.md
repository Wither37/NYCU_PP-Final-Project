# Experiments: how to run them and what you can change

Everything needed to produce the measurements and figures for the report.
Three scripts, each with one job, so experiments run once (on gpu7) and the
tables and figures can be rebuilt anywhere, as often as needed:

```
experiments/<name>.json      what to run (a "spec")
   │  scripts/sweep.py       runs bin/kkm once per configuration and repetition
   ▼
results/<name>/<run>.json    one report per run (+ <run>.log, kkm's output)
   │  scripts/collect.py     flattens the reports into a table
   ▼
results/<name>/runs.csv      one row per run
   │  scripts/plot.py        median over repetitions, one line per strategy
   ▼
plots/<figure>.pdf           figure for the report
```

`results/`, `plots/` and `*.csv` are git-ignored: never commit measurements,
copy the figures the report uses into the report directory instead.

## Quick start

On gpu7 (through `scripts/gpu7.sh`, which syncs, builds, and picks the
idlest GPU right before the run):

```bash
python3 scripts/sweep.py experiments/vs_n.json --dry-run          # check the run list first
scripts/gpu7.sh python3 scripts/sweep.py experiments/vs_n.json    # run it
```

Afterwards, anywhere with Python 3 and matplotlib (copy `results/` over first):

```bash
python3 scripts/collect.py results/vs_n
python3 scripts/plot.py results/vs_n/runs.csv --x n --filter p_budget_gb=24 \
    --logx --logy -o plots/vs_n_24g.pdf
```

On a laptop, to check that the pipeline works (small sizes, numbers are
**not** for the report):

```bash
python3 scripts/sweep.py experiments/smoke.json
python3 scripts/collect.py results/smoke
python3 scripts/plot.py results/smoke/runs.csv --x n --logx --logy -o plots/smoke.png
```

## The specs in this directory

| Spec | Measures | Runs | Where |
|---|---|---|---|
| `vs_n.json` | ms/iter vs n at d=50, k=10, for 24 and 48 GB budgets; n spans the in-memory limits (76k, 107k), the S1 limit (152k) and the worked example (215k) | 216 | gpu7 |
| `vs_d.json` | ms/iter vs d at n=40000, k=10; d covers Letter (26), Acoustic (50), the predicted crossovers (57, 737) and CIFAR-10 (3072) | 153 | gpu7 |
| `vs_k.json` | k = 10, 50, 100 at n=40000, d=50 | 27 | gpu7 |
| `cpu_threads.json` | OpenMP scaling, 1..64 threads, n=20000 | 21 | gpu7 (64 cores) |
| `smoke.json` | pipeline check; 0.25 GB budget so in-memory goes over budget at n=8000 | 24 | laptop |
| `cpu_threads_local.json` | OpenMP scaling on a laptop, n=8000 | 10 | laptop |

The n list, d list and repetition counts are first choices; edit them freely.
**When S3 exists, add `"s3"` to the `strategy` lists.**

## Writing or editing a spec

```json
{
  "name": "vs_n",
  "reps": 3,
  "timeout_s": 3600,
  "base": {"d": 50, "k": 10},
  "grid": {"budget_gb": [24, 48], "strategy": ["inmem", "s1", "s2"], "n": [10000, 20000]}
}
```

| Key | Meaning |
|---|---|
| `name` | output directory, `results/<name>/`. Use a new name for a new experiment. |
| `reps` | how many times each configuration runs (default 1). Use ≥ 3 for the report. |
| `timeout_s` | optional; a run longer than this is killed and recorded as `timeout`. |
| `base` | parameters every run shares. |
| `grid` | parameters that vary; every combination runs (Cartesian product). 2 × 3 × 2 above = 12 configurations, × `reps`. |

A parameter may sit in `base` or in `grid`, not both. The run name is built
from the `grid` parameters (`budget_gb-24_strategy-s2_n-20000_rep0`), so
moving a parameter between `base` and `grid` changes run names.

### Parameters (all map to `bin/kkm` flags)

| Parameter | kkm flag | Meaning | kkm default |
|---|---|---|---|
| `strategy` | `--strategy` | `cpu`, `inmem`, `s1`, `s2` (later `s3`) | `s2` |
| `n`, `d` | `--synthetic N D` | synthetic Gaussian blobs, n points in d dims (give both) | — |
| `input` | `--input` | a KKM1 binary dataset instead of `n`/`d` | — |
| `libsvm` | `--libsvm` | a libSVM text dataset instead of `n`/`d` (slow to parse; prefer `input`) | — |
| `k` | `--k` | clusters, 1..1024 | 10 |
| `iters` | `--iters` | iterations; keep 30 (matches Popcorn) | 30 |
| `seed` | `--seed` | data and initial-label seed; keep fixed so reps see identical input | 1 |
| `centers` | `--centers` | blob centers in the synthetic data | = k |
| `budget_gb` | `--budget-gb` | nominal GPU budget; 90% is usable (10% headroom) | 24 |
| `host_budget_gb` | `--host-budget-gb` | pinned host budget for S2, and the CPU path's K | 0.8 × MemAvailable |
| `tile` | `--tile` | tile edge for S1/S2 | 4096 |
| `nbuf` | `--nbuf` | S2 device buffers = CUDA streams | 3 |
| `threads` | `--threads` | OpenMP threads for `cpu` | all cores |

Turn a real dataset into a KKM1 file once, then sweep it with `input`:
`./bin/kkm --libsvm acoustic.txt --save-data data/acoustic.kkm1`
(`data/` is git-ignored).

## `scripts/sweep.py` options

| Option | Effect |
|---|---|
| `--dry-run` | print every kkm command and the run count; run nothing |
| (none) | run; finished runs (`ok` / `over_budget`) are **skipped**, so rerunning resumes an interrupted sweep |
| `--rerun-contended` | also redo finished runs that were flagged `contended` |
| `--force` | redo everything |
| `--kkm PATH` | binary to use (default `bin/kkm`) |
| `--out DIR` | parent of `<name>/` (default `results/`) |
| `--busy-pct P` | GPU utilization before a run at which it is flagged contended (default 10) |
| `--settle-s S` | pause before measuring the GPU load, so our previous run has left the sample (default 2) |

Behaviour worth knowing:

- **`over_budget` is a result, not a failure.** kkm exits 3 when a strategy
  cannot fit the budget; that is exactly where the vs_n plot draws its
  "over budget" line. The sweep exits non-zero only for `timeout`, `crash`
  or `error` runs.
- **Repetitions are interleaved**: every configuration once, then all of them
  again. Load on the shared machine drifts; this keeps a busy period from
  landing on all repetitions of one configuration.
- **Contention is recorded for every run**: GPU utilization and memory
  right before and after, host load before, and the git commit. A GPU run
  that starts with utilization ≥ `--busy-pct` is flagged `contended`;
  `plot.py` leaves those out. Rerun them with `--rerun-contended` once the
  GPU is idle. CPU runs are never flagged on GPU load.
- One run at a time on one GPU, so the course's one-GPU rule holds.

### Run status values

| Status | Meaning |
|---|---|
| `ok` | finished |
| `over_budget` | the strategy does not fit the GPU or host budget at this size (expected) |
| `timeout` | exceeded `timeout_s` |
| `error` | kkm reported an error (message in the report's `error` field) |
| `crash` | kkm died without a report (CUDA error, signal); see `<run>.log` |

## `scripts/collect.py` output: `runs.csv`

```bash
python3 scripts/collect.py results/vs_n                              # -> results/vs_n/runs.csv
python3 scripts/collect.py results/vs_d results/vs_k -o all.csv      # several sweeps into one table
```

| Column | Meaning |
|---|---|
| `p_<param>` | the parameters of the run, as given in the spec |
| `status`, `contended` | see above |
| `iter_ms_med`, `iter_ms_min` | time per iteration (S + assignment), median / min over iterations **2..N** |
| `iter_ms_first` | iteration 1 on its own (it carries warm-up cost) |
| `s_ms_med`, `s_ms_min` | time of `compute_S` alone (CUDA events), iterations 2..N |
| `setup_ms` | one-time construction (building K or the tiles; CPU: building K) |
| `upload_ms`, `total_ms` | upload of X; end-to-end time inside kkm |
| `device_peak`, `host_peak` | peak device / pinned host bytes actually allocated |
| `device_limit`, `host_limit`, `budget_bytes` | the limits in force |
| `objective_last` | final objective; equal across strategies up to rounding |
| `gpu_util_before`, `gpu_util_after`, `gpu_mem_used_before_mib`, `host_load_before` | load around the run |
| `wall_s`, `exit_code`, `git`, `started_utc` | bookkeeping |

## `scripts/plot.py`

| Option | Effect |
|---|---|
| `--x COL` | x axis: `n`, `d`, `k`, `threads`, … (required) |
| `--y COL` | y axis (default `iter_ms_med`; also `s_ms_med`, `setup_ms`, `total_ms`, `device_peak`) |
| `--series COL` | one line per value (default `strategy`) |
| `--filter COL=VALUE` | keep matching rows; repeatable. Numeric compare, so `p_budget_gb=24` matches `24.0` |
| `--logx`, `--logy` | log axes (ticks at 1-2-5, plain numbers) |
| `--speedup` | plot time(smallest x) / time, with an ideal line (for thread scaling) |
| `--keep-contended` | include contended runs |
| `--title`, `--xlabel`, `--ylabel` | override labels |
| `-o FILE` | output; `.pdf` for the report (vector), `.png` to look at |

Each point is the median over repetitions; the error bar spans min..max. A
dotted vertical line marks the first x at which a strategy went over budget.
A point must come from one configuration: if another sweep parameter varies
at the same x (e.g. `vs_n` has two budgets), `plot.py` stops and names the
`--filter` to add, rather than silently mixing them.
Each strategy keeps one color in every figure (in-memory blue, S1 orange,
S2 green, S3 yellow, CPU pink).

The report figures:

```bash
python3 scripts/plot.py results/vs_n/runs.csv --x n --filter p_budget_gb=24 --logx --logy -o plots/vs_n_24g.pdf
python3 scripts/plot.py results/vs_n/runs.csv --x n --filter p_budget_gb=48 --logx --logy -o plots/vs_n_48g.pdf
python3 scripts/plot.py results/vs_d/runs.csv --x d --logx --logy -o plots/vs_d.pdf
python3 scripts/plot.py results/vs_k/runs.csv --x k -o plots/vs_k.pdf
python3 scripts/plot.py results/cpu_threads/runs.csv --x threads --speedup --logx --logy -o plots/cpu_scaling.pdf
```

## Requirements

- `sweep.py`, `collect.py`: Python 3.8+, standard library only (runs on gpu7 as-is).
- `plot.py`: matplotlib.
- `bin/kkm` built for the GPU you run on (`make` on gpu7; see `CLAUDE.md`).
