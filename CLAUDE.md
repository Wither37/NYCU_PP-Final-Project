# NYCU PP-f26 — Memory-Scalable Kernel K-means on a Single GPU

Course project for Parallel Programming (NYCU, PP-f26). The goal is to run exact
Kernel K-means when the `n x n` kernel matrix `K` does not fit in GPU memory, on a
**single** GPU, and to quantify what each approach costs.

Team (sorted by student ID) and division of work:

| Member | Owns |
|---|---|
| 112550009 賴邑城 | S3 fused recomputation kernel, roofline analysis |
| 112550013 周廷威 | S1 and S2, Nsight timeline analysis |
| 112550205 張程翔 | shared framework, OpenMP CPU version, Popcorn baseline, selector + cost model, experiment scripts and plots |

Course site: <https://nycu-sslab.github.io/PP-f26/project/>

## Course deadlines (from the course project page)

| Date | Item |
|---|---|
| Sep 24, 2026 | Group registration (done) |
| **Oct 29, 2026** | **Project proposal due** |
| Weeks 13–16 | Final project presentations; our slot is not yet confirmed |
| 23:59 the day before our talk | Presentation slides due |
| **Dec 28, 2026** | **Final report and source code due** |

Everything is submitted to e-Campus. Late penalty is 15% per day, and weekends
count as a single day.

### What the course requires of each artefact

- **Proposal** — ACM sigconf, the 10 sections in the fixed order we already
  follow. English or Chinese both allowed.
- **Final report** — ACM sigconf, **4 to 8 pages, 9-point font**. Sections:
  title, participants, abstract (≤100 words), introduction, proposed solution,
  how parallelization is done, experimental methodology, experimental results,
  related work, **contributions of each member**, conclusions, references.
- **Presentation** — 10 min talk + 3 min Q&A, strictly timed. Must cover
  changes since the proposal, how parallelism was actually implemented (not
  just which model was used), challenges encountered, and contributions of each
  member. Attendance at all four weeks of presentations is mandatory; missing a
  peer review costs 1 point each, and at least one Slido question across the
  four weeks is required or it is a 5-point deduction.

## Repository layout

```
proposal/
  main.tex          proposal source (ACM sigconf, XeLaTeX)
  refs.bib          bibliography -- 8 entries, all verified; do not add unverified ones
  figures/arch.tex  TikZ block diagram, \input from main.tex
  Makefile          `make` -> main.pdf + Proposal_6.pdf ; `make clean` ; `make purge`
                    submission filename rule is Proposal_<Group_id>.pdf; we are
                    group 6, override with `make GROUP=06`
  .latexmkrc        forces XeLaTeX (pdf_mode = 5)
  hardware.md       hardware report + microbenchmark results + all derived arithmetic
  hwbench/          CUDA microbenchmarks (devinfo, bw_pcie, bw_gpu) + run.sh
```

## Building the proposal

**XeLaTeX is mandatory** — the author names are Chinese and are typeset with
xeCJK + the Noto CJK TC fonts. pdfLaTeX will not compile this document.

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
cd proposal && make
```

TeX Live 2026 is installed per-user via TinyTeX at `~/.TinyTeX` (there is no
root on this machine and no system TeX). If a package is missing:
`tlmgr install <pkg>`.

Before declaring a build good, check `proposal/main.log` for:
`Overfull \hbox`, `LaTeX Warning: Citation`, `LaTeX Warning: Reference`.
The current build has **zero** of each and is 4 pages; keep it that way.

## Hardware: `gpu7.cplab.cs.nycu.edu.tw`

Shared server. **10x RTX A6000 (48 GB, cc 8.6, 84 SMs, PCIe Gen4 x16)**,
2x Xeon Gold 6448Y (64c/128t), 1007 GiB RAM. CUDA toolkit 12.4 at
`/usr/local/cuda-12.4`; compile with `-arch=sm_86`.

### Shared-machine etiquette — follow this every time

Other students run long jobs on most GPUs. Before any GPU work:

1. Run `nvidia-smi`, pick the GPU with the **lowest utilization** and most free
   memory, and set `CUDA_VISIBLE_DEVICES` to it. `proposal/hwbench/run.sh` does
   this automatically.
2. **Re-check immediately before the run**, not just once — a GPU that was idle
   during the survey can be claimed seconds later. This already happened once:
   GPU 0 was taken mid-session and the device-bandwidth numbers dropped from
   679 GB/s to as low as 444 GB/s. We re-ran on an idle GPU.
3. Keep exploratory runs short and the allocation small. Free everything
   (`cudaDeviceReset()`) before exiting.
4. Record the foreign utilization that was present during any measurement, and
   say so in writing. A number taken under contention is not a measurement.

## Project invariants

- **Never derive limits from observed free GPU memory.** It depends on other
  users. Use a configurable budget with 10% headroom: `M_gpu = 0.9 * M_budget`,
  evaluated at both 24 GB and 48 GB. The implementation must enforce this budget
  so experiments are reproducible. Host budget is `0.8 *` *available* RAM.
- **Hand-written CUDA only.** The course forbids relying on libraries that
  supply the parallelism. cuSPARSE / cuBLAS / RAFT may appear *only* inside the
  Popcorn baseline we compare against.
- **Do not invent numbers.** Measured, derived and assumed quantities are
  labelled separately in `hardware.md`. Today `B_pcie = 26.25 GB/s` and
  `B_gpu = 679.35 GB/s` are measured; `F_eff = 0.5 * F_peak = 19.35 TFLOP/s` is
  an **assumption** because the fused kernel does not exist yet. Replace it with
  a measurement as soon as the kernel runs, then update every number in
  `hardware.md` Section 4 and in `main.tex` Section 6 that depends on it.
- No algorithm results exist yet. Everything in the proposal is phrased as
  *expected* or *predicted*; keep it that way until there are real timings.
- Fixed 30 iterations per clustering run, matching the Popcorn evaluation, so
  timings compare like with like.
- Core deliverables are the shared framework, the CPU reference, S2 and S3.
  **S1 and the automatic selector are stretch goals**, and the proposal says so.
- Bellavita et al. 2026 (arXiv:2601.17136), "Communication-Avoiding Linear
  Algebraic Kernel K-Means on GPUs", is the Popcorn authors' own follow-up: it
  solves the same memory limit with **multi-GPU** distributed algorithms and
  mentions a single-GPU sliding-window baseline that recomputes kernel blocks.
  Our niche is the single-GPU case; cite and position against it everywhere.

## The four strategies (all compute the same `S`, `n x k`)

| | storage for `K` | bound by |
|---|---|---|
| In-memory | `4n^2` on GPU | GPU memory |
| S1 symmetric tiles | `2n^2` on GPU | GPU memory (only a `sqrt(2)` gain); **stretch goal** |
| S2 out-of-core | `2n^2` in pinned host RAM, streamed | PCIe bandwidth |
| S3 fused recompute | none; `O(nd + nk)` total | FP32 throughput |

Cost model: `T_in-mem ~ 4n^2/B_gpu`, `T_S2 ~ 2n^2/B_pcie`, `T_S3 ~ 2n^2 d/F_eff`.
Crossovers: S3 beats S2 when `d < F_eff/B_pcie = 737`; S3 beats in-memory when
`d < 2*F_eff/B_gpu = 57`. The `d < 57` crossover is the interesting claim — it
sits among real datasets (Acoustic `d=50`, Letter `d=26` as reported by Popcorn,
both below; MNIST `d=780` above).

## Conventions

- Capacities use `1 GB = 2^30 B`; bandwidths use decimal `GB/s` as measured.
  State which one you mean.
- `\todo{...}` renders red in the PDF. Use it for anything unresolved rather
  than silently guessing, and list the open ones when reporting a build.
- The proposal must keep its 10 required sections in the course-mandated order.
