#!/usr/bin/env python3
"""Plot one measure of a sweep against one parameter, a line per strategy.

    python3 scripts/plot.py results/vs_n/runs.csv --x n --y iter_ms_med \
        --filter p_budget_gb=24 --logx --logy -o plots/vs_n_24g.pdf
    python3 scripts/plot.py results/cpu_threads/runs.csv --x threads \
        --y iter_ms_med --speedup -o plots/cpu_scaling.pdf

Each point is the median over repetitions; the error bar spans their min..max.
Only status "ok" runs enter the lines, and runs flagged contended (the GPU was
busy right before them) are dropped unless --keep-contended. Where a series
went over budget, a dotted vertical line marks the first such x: that is where
the strategy stops being able to run at all.

The output format follows the extension (.pdf for the report, .png to look at).
Needs matplotlib; no pandas.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# Validated categorical palette (light surface), fixed slot per strategy so a
# strategy keeps its color in every figure of the report.
SERIES_STYLE = {
    "inmem": ("In-memory (full K)", "#2a78d6", "o"),
    "s1": ("S1 symmetric tiles", "#eb6834", "s"),
    "s2": ("S2 out-of-core", "#1baf7a", "^"),
    "s3": ("S3 fused recompute", "#eda100", "D"),
    "cpu": ("CPU (OpenMP)", "#e87ba4", "v"),
    "popcorn": ("Popcorn", "#008300", "P"),
}
SPARE = ["#6250d6", "#e34948"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

AXIS_LABELS = {
    "n": "n (points)", "d": "d (dimensions)", "k": "k (clusters)",
    "threads": "OpenMP threads",
    "iter_ms_med": "time per iteration (ms, median)",
    "s_ms_med": "time to compute S (ms, median)",
    "setup_ms": "construction time (ms)", "total_ms": "end-to-end time (ms)",
    "device_peak": "peak device memory (bytes)", "host_peak": "peak pinned host memory (bytes)",
}


def num(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load(path: Path, filters: list[str], keep_contended: bool):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for flt in filters:
        key, _, want = flt.partition("=")
        if rows and key not in rows[0]:
            sys.exit(f"--filter: no column '{key}'; columns: {', '.join(rows[0])}")
        target = num(want)  # numeric compare, so "24" matches "24.0"
        rows = [r for r in rows
                if (num(r[key]) == target if target is not None else r[key] == want)]
    dropped = [r for r in rows if r.get("contended") == "True" and r["status"] == "ok"]
    if dropped and not keep_contended:
        print(f"dropping {len(dropped)} contended run(s)", file=sys.stderr)
        rows = [r for r in rows if r not in dropped]
    failed = [r for r in rows if r["status"] not in ("ok", "over_budget")]
    if failed:
        print(f"ignoring {len(failed)} run(s) with status "
              + ", ".join(sorted({r['status'] for r in failed})), file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path, help="runs.csv from scripts/collect.py")
    ap.add_argument("--x", required=True, help="column on the x axis, e.g. n, d, k, threads")
    ap.add_argument("--y", default="iter_ms_med", help="column on the y axis (default %(default)s)")
    ap.add_argument("--series", default="strategy", help="one line per value (default %(default)s)")
    ap.add_argument("--filter", action="append", default=[], metavar="COL=VALUE",
                    help="keep only matching rows; repeatable, e.g. p_budget_gb=24")
    ap.add_argument("--speedup", action="store_true",
                    help="plot speedup relative to each series' smallest x (time / time)")
    ap.add_argument("--logx", action="store_true")
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--keep-contended", action="store_true")
    ap.add_argument("--title")
    ap.add_argument("--xlabel")
    ap.add_argument("--ylabel")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    rows = load(args.csv, args.filter, args.keep_contended)
    for col in (args.x, args.y, args.series):
        if rows and col not in rows[0]:
            sys.exit(f"no column '{col}' in {args.csv}")

    # Every point must aggregate repetitions of ONE configuration. If another
    # sweep parameter varies at the same (series, x) -- e.g. vs_n without a
    # budget filter -- medians would mix configurations and an over-budget run
    # at one budget would hide a valid point at the other. Refuse instead.
    own = {args.x, args.series} | {"p_" + c for c in (args.x, args.series)}
    own |= {c[2:] for c in own if c.startswith("p_")}
    pcols = [c for c in (rows[0] if rows else {}) if c.startswith("p_") and c not in own]
    seen = defaultdict(set)
    for r in rows:
        seen[(r[args.series], r[args.x])].add(tuple(r[c] for c in pcols))
    mixed = sorted({c for combos in seen.values() if len(combos) > 1
                    for i, c in enumerate(pcols) if len({t[i] for t in combos}) > 1})
    if mixed:
        sys.exit("several configurations share one point; pick one with --filter, e.g. "
                 + " ".join(f"--filter {c}=<value>" for c in mixed))

    points = defaultdict(lambda: defaultdict(list))  # series -> x -> [y]
    over = defaultdict(set)                          # series -> {x over budget}
    for r in rows:
        x = num(r[args.x])
        if x is None:
            continue
        if r["status"] == "over_budget":
            over[r[args.series]].add(x)
        elif r["status"] == "ok" and num(r[args.y]) is not None:
            points[r[args.series]][x].append(num(r[args.y]))

    names = [s for s in SERIES_STYLE if s in points or s in over]
    names += sorted((set(points) | set(over)) - set(names))
    if not names:
        sys.exit("nothing to plot after filtering")
    spare = iter(SPARE)
    style = {s: SERIES_STYLE.get(s) or (s, next(spare, TEXT_2), "o") for s in names}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT})
    fig, ax = plt.subplots(figsize=(4.6, 3.1), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    direct_label = len(names) <= 4
    for s in names:
        label, color, marker = style[s]
        xs = sorted(x for x in points.get(s, {}) if x not in over[s])
        if xs:
            meds = [statistics.median(points[s][x]) for x in xs]
            lo = [min(points[s][x]) for x in xs]
            hi = [max(points[s][x]) for x in xs]
            if args.speedup:
                base = meds[0]
                meds, lo, hi = ([base / v for v in arr] for arr in (meds, hi, lo))
            err = [[m - a for m, a in zip(meds, lo)], [b - m for m, b in zip(meds, hi)]]
            ax.errorbar(xs, meds, yerr=err, color=color, marker=marker, markersize=4.5,
                        linewidth=1.4, capsize=2, elinewidth=0.8, label=label, zorder=3,
                        markeredgecolor=SURFACE, markeredgewidth=0.8)
            if direct_label:
                ax.annotate(label, (xs[-1], meds[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=TEXT)
        if over[s]:
            x0 = min(over[s])
            ax.axvline(x0, color=color, linestyle=":", linewidth=1.2, zorder=2)
            # Vertical, along the line's left side at the bottom: clear of the
            # direct labels, which sit at the right end of each series.
            ax.annotate(f"{label.split(' (')[0]} over budget", (x0, 0.0),
                        xycoords=("data", "axes fraction"), xytext=(-3, 4),
                        textcoords="offset points", fontsize=7, color=TEXT_2,
                        rotation=90, ha="right", va="bottom")

    if args.speedup:
        xs_all = sorted({x for s in points for x in points[s]})
        if xs_all:
            ax.plot(xs_all, [x / xs_all[0] for x in xs_all], color=TEXT_2, linestyle="--",
                    linewidth=0.9, label="ideal", zorder=1)
    # Plain numbers (2,000 / 0.5) on log axes instead of 2x10^3, ticks at 1-2-5.
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    for on, axis, setter in ((args.logx, ax.xaxis, ax.set_xscale),
                             (args.logy, ax.yaxis, ax.set_yscale)):
        if on:
            setter("log")
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(plain)
            axis.set_minor_formatter(NullFormatter())
    ax.margins(x=0.18 if direct_label else 0.05)
    ax.set_xlabel(args.xlabel or AXIS_LABELS.get(args.x, args.x))
    if args.speedup:
        x0 = min(x for s in points for x in points[s])
        ylabel = f"speedup over {args.x} = {x0:g}"
    else:
        ylabel = AXIS_LABELS.get(args.y, args.y)
    ax.set_ylabel(args.ylabel or ylabel)
    if args.title:
        ax.set_title(args.title, fontsize=9.5, loc="left", color=TEXT)
    if len(names) >= 2 or args.speedup:
        ax.legend(frameon=False, fontsize=7.5, loc="best")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
