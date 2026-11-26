#!/usr/bin/env python3
"""Run a parameter sweep of bin/kkm, one JSON report per run.

    python3 scripts/sweep.py experiments/vs_n.json             # run (resumable)
    python3 scripts/sweep.py experiments/vs_n.json --dry-run   # list the runs
    scripts/gpu7.sh python3 scripts/sweep.py experiments/vs_n.json

Spec (JSON):

    {
      "name": "vs_n",                    # reports go to results/<name>/
      "reps": 3,                         # repetitions of every configuration
      "timeout_s": 3600,                 # optional, per run
      "base": {"d": 50, "k": 10},        # parameters shared by every run
      "grid": {"strategy": ["inmem", "s2"], "n": [10000, 20000]}
    }

`grid` is expanded as a Cartesian product. Parameters: strategy, n and d
(synthetic data), input (KKM1 file) or libsvm (text file) instead of n/d, k,
iters, seed, centers, budget_gb, host_budget_gb, tile, nbuf, threads.

Each run writes results/<name>/<run_id>.json (kkm's own report plus a "meta"
object: parameters, exit code, wall time, GPU and host load right before and
after the run) and <run_id>.log (kkm's stdout/stderr). Runs whose report
already says "ok" or "over_budget" are skipped, so an interrupted sweep
resumes where it stopped; --rerun-contended also redoes the runs that were
flagged contended, --force reruns everything. See experiments/README.md.

Repetitions are interleaved (all configurations once, then all again) so that
a slow drift on a shared machine does not land on one configuration.

Exit code 3 from kkm ("over_budget") is an expected result, not a failure.
Python 3.8+ standard library only, so it runs on gpu7 as-is.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Spec parameter -> kkm flag. n and d are handled together (--synthetic N D).
FLAGS = {
    "strategy": "--strategy",
    "k": "--k",
    "iters": "--iters",
    "seed": "--seed",
    "centers": "--centers",
    "budget_gb": "--budget-gb",
    "host_budget_gb": "--host-budget-gb",
    "tile": "--tile",
    "nbuf": "--nbuf",
    "threads": "--threads",
    "input": "--input",
    "libsvm": "--libsvm",
}
KNOWN = set(FLAGS) | {"n", "d"}
DONE = {"ok", "over_budget"}
OVER_BUDGET_EXIT = 3


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    for key in ("name", "grid"):
        if key not in spec:
            sys.exit(f"{path}: missing '{key}'")
    spec.setdefault("reps", 1)
    spec.setdefault("base", {})
    unknown = (set(spec["base"]) | set(spec["grid"])) - KNOWN
    if unknown:
        sys.exit(f"{path}: unknown parameter(s) {sorted(unknown)}; known: {sorted(KNOWN)}")
    both = set(spec["base"]) & set(spec["grid"])
    if both:
        sys.exit(f"{path}: {sorted(both)} in both base and grid; keep each in one")
    for key, values in spec["grid"].items():
        if not isinstance(values, list) or not values:
            sys.exit(f"{path}: grid '{key}' must be a non-empty list")
    return spec


def expand(spec: dict) -> list[dict]:
    keys = list(spec["grid"])
    configs = []
    for combo in itertools.product(*(spec["grid"][k] for k in keys)):
        cfg = dict(spec["base"])
        cfg.update(zip(keys, combo))
        configs.append(cfg)
    return configs


def run_id(cfg: dict, grid_keys: list[str], rep: int) -> str:
    """Stable, filesystem-safe name built from the parameters that vary."""
    parts = []
    for k in grid_keys:
        v = cfg[k]
        if k in ("input", "libsvm"):
            v = Path(str(v)).stem
        parts.append(f"{k}-{v}")
    parts.append(f"rep{rep}")
    return "_".join(parts).replace("/", "-").replace(" ", "")


def kkm_command(kkm: str, cfg: dict, json_path: Path) -> list[str]:
    cmd = [kkm]
    if "input" not in cfg and "libsvm" not in cfg:
        if "n" not in cfg or "d" not in cfg:
            sys.exit(f"config {cfg} needs n and d (or input / libsvm)")
        cmd += ["--synthetic", str(cfg["n"]), str(cfg["d"])]
    for key, flag in FLAGS.items():
        if key in cfg:
            cmd += [flag, str(cfg[key])]
    cmd += ["--json", str(json_path)]
    return cmd


def gpu_probe() -> dict | None:
    """Utilization and memory of the GPU this process will use, or None."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or "0"
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", dev,
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.strip().splitlines()
    fields = [f.strip() for f in lines[0].split(",")] if lines else []
    if len(fields) != 5:
        return None

    def num(s):
        try:
            return float(s)
        except ValueError:
            return None

    return {"index": fields[0], "name": fields[1], "util_pct": num(fields[2]),
            "mem_used_mib": num(fields[3]), "mem_total_mib": num(fields[4])}


def host_load() -> list[float] | None:
    try:
        return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except OSError:
        return None


def git_commit() -> str:
    # gpu7.sh rsyncs without .git, so it passes the commit in KKM_GIT instead.
    if os.environ.get("KKM_GIT"):
        return os.environ["KKM_GIT"]
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def is_done(path: Path, rerun_contended: bool) -> bool:
    """True when an earlier report for this run can be kept."""
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if rep.get("status") not in DONE:
        return False
    return not (rerun_contended and rep.get("meta", {}).get("contended"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", type=Path)
    ap.add_argument("--kkm", default=str(ROOT / "bin" / "kkm"))
    ap.add_argument("--out", type=Path, default=ROOT / "results",
                    help="parent directory for results/<name>/ (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="print the runs, run nothing")
    ap.add_argument("--force", action="store_true", help="rerun finished runs too")
    ap.add_argument("--rerun-contended", action="store_true",
                    help="rerun only the finished runs that were flagged contended")
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="pause before probing the GPU, so our previous run has "
                         "left the utilization sample (default: %(default)s)")
    ap.add_argument("--busy-pct", type=float, default=10.0,
                    help="GPU utilization before a run at or above which the run is "
                         "flagged contended (default: %(default)s)")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    configs = expand(spec)
    grid_keys = list(spec["grid"])
    outdir = args.out / spec["name"]
    timeout = spec.get("timeout_s")
    runs = [(rep, cfg) for rep in range(spec["reps"]) for cfg in configs]
    print(f"# sweep {spec['name']}: {len(configs)} configurations x {spec['reps']} reps "
          f"= {len(runs)} runs -> {outdir}")

    if args.dry_run:
        for rep, cfg in runs:
            rid = run_id(cfg, grid_keys, rep)
            print(shlex.join(kkm_command(args.kkm, cfg, outdir / f"{rid}.json")))
        return 0

    if not Path(args.kkm).exists():
        sys.exit(f"{args.kkm} not found -- build first (make)")
    outdir.mkdir(parents=True, exist_ok=True)
    commit, host = git_commit(), socket.gethostname()
    tally: dict[str, int] = {}

    for i, (rep, cfg) in enumerate(runs, 1):
        rid = run_id(cfg, grid_keys, rep)
        jpath, lpath = outdir / f"{rid}.json", outdir / f"{rid}.log"
        if not args.force and is_done(jpath, args.rerun_contended):
            tally["skipped"] = tally.get("skipped", 0) + 1
            continue
        jpath.unlink(missing_ok=True)  # never mistake a stale report for this run's
        cmd = kkm_command(args.kkm, cfg, jpath)

        time.sleep(args.settle_s)
        before, load_before = gpu_probe(), host_load()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.monotonic()
        with open(lpath, "w", encoding="utf-8") as log:
            log.write("$ " + shlex.join(cmd) + "\n")
            log.flush()
            try:
                exit_code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                exit_code = None
        wall_s = time.monotonic() - t0
        after = gpu_probe()

        try:
            report = json.loads(jpath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}  # kkm died before writing (CUDA error, signal, timeout)
        if exit_code is None:
            report["status"] = "timeout"
        elif "status" not in report:
            report["status"] = "crash"
        elif exit_code == OVER_BUDGET_EXIT:
            report["status"] = "over_budget"

        # GPU contention only matters for GPU strategies; a CPU run records the
        # host load instead (on a laptop the display alone keeps the GPU busy).
        util = before["util_pct"] if before and cfg.get("strategy") != "cpu" else None
        report["meta"] = {
            "sweep": spec["name"], "run_id": rid, "rep": rep, "params": cfg,
            "cmd": shlex.join(cmd), "exit_code": exit_code, "wall_s": round(wall_s, 3),
            "started_utc": started, "host": host, "git": commit,
            "gpu_before": before, "gpu_after": after, "host_load_before": load_before,
            "contended": util is not None and util >= args.busy_pct,
        }
        jpath.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

        status = report["status"]
        tally[status] = tally.get(status, 0) + 1
        flag = "  [CONTENDED: GPU busy before the run]" if report["meta"]["contended"] else ""
        print(f"[{i}/{len(runs)}] {rid}: {status} ({wall_s:.1f} s){flag}", flush=True)

    print("# done: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    bad = sum(v for k, v in tally.items() if k not in DONE and k != "skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
