#!/usr/bin/env python3
"""Flatten the per-run JSON reports of a sweep into one CSV row per run.

    python3 scripts/collect.py results/vs_n          # -> results/vs_n/runs.csv

Columns: the sweep parameters, status, and the timing/memory summary of each
run. Per-iteration times are summarized over iterations 2..N; the first
iteration is reported separately (iter_ms_first) because it carries warm-up
costs that the steady-state comparison should not include.

    *_med      median over iterations 2..N
    *_min      minimum over iterations 2..N
    setup_ms   one-time construction (GPU: strategy setup; CPU: building K)
    total_ms   end-to-end wall time measured inside kkm

Repetitions stay separate rows; scripts/plot.py aggregates them.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

SUMMARY = ["status", "contended", "strategy", "n", "d", "k", "iters", "seed",
           "iter_ms_med", "iter_ms_min", "iter_ms_first", "s_ms_med", "s_ms_min",
           "setup_ms", "upload_ms", "total_ms",
           "device_peak", "device_limit", "host_peak", "host_limit",
           "budget_bytes", "host_budget_bytes", "threads",
           "objective_last", "check_label_agreement", "check_objective_rel",
           "gpu", "gpu_util_before", "gpu_util_after", "gpu_mem_used_before_mib",
           "host_load_before", "wall_s", "exit_code", "git", "started_utc", "error"]


def steady(values) -> list[float]:
    vals = [v for v in (values or []) if v is not None]
    return vals[1:] if len(vals) > 1 else vals


def summarize(rep: dict) -> dict:
    meta = rep.get("meta", {})
    params = meta.get("params", {})
    before, after = meta.get("gpu_before") or {}, meta.get("gpu_after") or {}
    it, s = steady(rep.get("iter_ms")), steady(rep.get("s_ms"))
    objective = [v for v in rep.get("objective", []) if v is not None]
    row = {f"p_{k}": v for k, v in params.items()}
    row.update({
        "run_id": meta.get("run_id"), "rep": meta.get("rep"),
        "status": rep.get("status"), "contended": meta.get("contended"),
        "strategy": rep.get("strategy", params.get("strategy")),
        "n": rep.get("n", params.get("n")), "d": rep.get("d", params.get("d")),
        "k": rep.get("k", params.get("k")),
        "iters": rep.get("iters"), "seed": rep.get("seed"),
        "iter_ms_med": statistics.median(it) if it else None,
        "iter_ms_min": min(it) if it else None,
        "iter_ms_first": (rep.get("iter_ms") or [None])[0],
        "s_ms_med": statistics.median(s) if s else None,
        "s_ms_min": min(s) if s else None,
        # The CPU path calls its construction build_ms; GPU strategies setup_ms.
        "setup_ms": rep.get("setup_ms", rep.get("build_ms")),
        "upload_ms": rep.get("upload_ms"), "total_ms": rep.get("total_ms"),
        "objective_last": objective[-1] if objective else None,
        "gpu_util_before": before.get("util_pct"),
        "gpu_util_after": after.get("util_pct"),
        "gpu_mem_used_before_mib": before.get("mem_used_mib"),
        "host_load_before": (meta.get("host_load_before") or [None])[0],
    })
    for key in ("device_peak", "device_limit", "host_peak", "host_limit", "budget_bytes",
                "host_budget_bytes", "threads", "gpu", "check_label_agreement",
                "check_objective_rel", "error"):
        row[key] = rep.get(key)
    for key in ("wall_s", "exit_code", "git", "started_utc"):
        row[key] = meta.get(key)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path, help="results/<sweep> directories")
    ap.add_argument("-o", "--output", type=Path,
                    help="CSV path (default: <dir>/runs.csv; required for several dirs)")
    args = ap.parse_args()
    if len(args.dirs) > 1 and not args.output:
        sys.exit("several directories: give -o")

    rows = []
    for d in args.dirs:
        for path in sorted(d.glob("*.json")):
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as e:
                print(f"skipping {path}: {e}", file=sys.stderr)
                continue
            if "meta" not in rep:
                print(f"skipping {path}: not written by sweep.py", file=sys.stderr)
                continue
            rows.append(summarize(rep))
    if not rows:
        sys.exit("no reports found")

    params = sorted({k for r in rows for k in r if k.startswith("p_")})
    header = ["run_id", "rep"] + params + SUMMARY
    out = args.output or args.dirs[0] / "runs.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in header})

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    contended = sum(1 for r in rows if r.get("contended"))
    print(f"{out}: {len(rows)} runs (" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + (f"; {contended} contended" if contended else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
