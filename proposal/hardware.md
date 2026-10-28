# Hardware report — `gpu7.cplab.cs.nycu.edu.tw`

Collected 2026-10-22 for the NYCU PP-f26 project *Memory-Scalable Kernel K-means on a
Single GPU*. This is a **shared** server: 10× NVIDIA RTX A6000 and ~1 TB of host RAM are
used concurrently by other students' jobs, so every number below is annotated with the
contention present when it was taken.

## 1. GPU selection

`nvidia-smi` at 22:43:33 (2026-10-22) reported all ten GPUs:

| GPU | Bus Id | Memory used / total (MiB) | Util | Power |
|----:|--------|--------------------------:|-----:|------:|
| **0** | `0000:2E:00.0` | **4084 / 49140** | **0 %** | 80 W / 300 W |
| 1 | `0000:30:00.0` | 39661 / 49140 | 100 % | 291 W |
| 2 | `0000:3F:00.0` | 39661 / 49140 | 100 % | 292 W |
| 3 | `0000:40:00.0` | 9151 / 49140 | 41 % | 161 W |
| 4 | `0000:41:00.0` | 8099 / 49140 | 100 % | 271 W |
| 5 | `0000:B0:00.0` | 7191 / 49140 | 87 % | 279 W |
| 6 | `0000:B1:00.0` | 29711 / 49140 | 100 % | 283 W |
| 7 | `0000:C1:00.0` | 32067 / 49140 | 99 % | 276 W |
| 8 | `0000:C2:00.0` | 33221 / 49140 | 100 % | 298 W |
| 9 | `0000:C3:00.0` | 19297 / 49140 | 100 % | 172 W |

**GPU 0 was selected** — the only idle device (0 % utilization) and the one with the most
free memory (≈45 GB). All work below ran with `CUDA_VISIBLE_DEVICES=0`. The 4 GB resident
on GPU 0 belongs to another user's Python process (PID 1993117) plus Xorg; it was idle
throughout.

## 2. Platform

| Property | Value |
|---|---|
| GPU model | NVIDIA RTX A6000 (Ampere, GA102) |
| Total memory | 49140 MiB ≈ 48 GB (48539 MiB usable per the runtime) |
| Memory bus width | 384 bit @ 8001 MHz → 768.1 GB/s theoretical |
| PCIe link | **Gen 4 × 16** (device max Gen 4, host max Gen 5, current width 16x) |
| Driver version | 580.105.08 |
| CUDA (driver) | 13.0 |
| CUDA toolkit (`nvcc`) | **12.4**, V12.4.131 — at `/usr/local/cuda-12.4/bin/nvcc` |
| Compute capability | **8.6** (`-arch=sm_86`) |
| ECC | disabled |

### `cudaGetDeviceProperties` (via [`hwbench/devinfo.cu`](hwbench/devinfo.cu))

```
name                 : NVIDIA RTX A6000
compute capability   : 8.6
SM count             : 84
FP32 cores per SM    : 128      (cc 8.6, GA10x)
boost clock          : 1.800 GHz
memory clock         : 8001 MHz
memory bus width     : 384 bit
total global memory  : 48539 MiB
L2 cache             : 6144 KiB
shared mem per block : 49152 B
shared mem per SM    : 102400 B
warp size            : 32
async engine count   : 2        (enough for copy/compute overlap in S2)
ECC enabled          : 0
theoretical FP32 peak: 38.71 TFLOP/s
theoretical mem BW   : 768.1 GB/s
```

### Host

| Property | Value |
|---|---|
| CPU | 2 × Intel Xeon Gold 6448Y |
| Cores / threads | 64 physical cores (32 per socket, 2 sockets), 128 hardware threads |
| Clock | 0.8–4.1 GHz |
| Caches | L2 128 MiB total, L3 120 MiB (2 × 60 MiB) |
| NUMA | 2 nodes (node0 = CPU 0–31,64–95; node1 = 32–63,96–127) |
| Total RAM | 1007 GiB (`free -g`) |
| **Available RAM** | **705 GiB** at the time of measurement |
| Swap | 31 GiB (30 GiB already in use — host memory is under real pressure) |

A usable NVIDIA GPU is present, so we proceed to the microbenchmarks.

---

## 3. Microbenchmarks

Code: [`hwbench/bw_pcie.cu`](hwbench/bw_pcie.cu), [`hwbench/bw_gpu.cu`](hwbench/bw_gpu.cu),
driver script [`hwbench/run.sh`](hwbench/run.sh) (`make run`). Each benchmark does one
untimed warm-up and then 5 timed trials with CUDA events. Peak allocation is 1 GiB pinned
host + 1 GiB device for `bw_pcie` and 2 GiB device for `bw_gpu` — well inside the 4 GB
cap — and both call `cudaDeviceReset()` before exiting so nothing is left resident. Total
wall time for all runs was **under 3 seconds**.

### Contention during the runs — important

GPU 0, idle when first surveyed, was claimed by another user's job between the survey and
the benchmark run. The first attempt therefore ran at **60–95 % foreign utilization** and
produced depressed and noisy device bandwidth (443–620 GB/s). Following the
"re-check and switch" rule, we re-surveyed, found **GPU 9** at 0 % utilization, and re-ran
there. The reported numbers below are the **GPU 9** run, where foreign utilization was
**0 %** at launch and the only foreign resident memory was 4197 MiB belonging to an idle
process. (A third user's job appeared on GPU 9 immediately after our run finished; it did
not overlap the measurements.) Both runs are kept here because the contrast quantifies
how much a shared GPU perturbs measurements.

### B_pcie — host-to-device bandwidth, pinned memory, 1 GiB per copy

| Trial | H2D (GB/s) | D2H (GB/s) |
|---|---:|---:|
| 0 | 26.04 | 26.39 |
| 1 | 26.24 | 26.39 |
| 2 | 25.81 | 26.40 |
| 3 | 26.40 | 26.40 |
| 4 | 26.25 | 26.40 |
| **median** | **26.25** | **26.39** |
| **max** | **26.40** | **26.40** |

We use **B_pcie = 26.25 GB/s** (H2D median). That is 83 % of the 31.5 GB/s nominal PCIe
Gen 4 ×16 payload rate, which is the usual efficiency for pinned `cudaMemcpy`, and it
confirms the link is really running at Gen 4 ×16.

### B_gpu — device memory bandwidth, read + write copy kernel

2 GiB moved per trial (1 GiB read + 1 GiB written), counted as read + write bytes.

| Trial | Time (ms) | B_gpu (GB/s) |
|---|---:|---:|
| 0 | 3.163 | 678.91 |
| 1 | 3.156 | 680.45 |
| 2 | 3.162 | 679.13 |
| 3 | 3.161 | 679.35 |
| 4 | 3.159 | 679.79 |
| **median** | 3.161 | **679.35** |
| **max** | 3.156 | **680.45** |

We use **B_gpu = 679.35 GB/s** (median), i.e. **88.4 %** of the 768.1 GB/s theoretical
peak — a normal achievable fraction for a streaming copy kernel on GDDR6.

(Contended first attempt on GPU 0, for reference: median 617.54 GB/s, max 620.09 GB/s,
with one trial as low as 443.65 GB/s.)

### F — theoretical FP32 peak

Compute capability 8.6 (GA102) has **128 FP32 cores per SM**.

```
F_peak = SMs x cores/SM x 2 x boost clock
       = 84 x 128 x 2 x 1.800e9
       = 38.7072e12 FLOP/s = 38.71 TFLOP/s
```

**Assumption, stated explicitly:** our fused S3 kernel does not exist yet, so we cannot
measure its achieved throughput. For every projection in this report we assume it reaches
half of peak:

```
F_eff = 0.5 x F_peak = 19.3536e12 FLOP/s = 19.35 TFLOP/s
```

This is a *planning assumption*, not a measurement. A tiled kernel that applies a kernel
function between shared-memory tiles will not hit the FMA peak that a pure SGEMM reaches;
50 % is optimistic-but-plausible for a hand-written tiled kernel and must be replaced by a
measurement once the kernel exists.

### Summary of constants used below

| Symbol | Value | Status |
|---|---:|---|
| B_pcie | 26.25 GB/s | **measured** (median, pinned H2D) |
| B_gpu | 679.35 GB/s | **measured** (median, read+write kernel) |
| F_peak | 38.71 TFLOP/s | derived from device properties |
| F_eff | 19.35 TFLOP/s | **assumed** = 0.5 × F_peak |
| B_gpu / B_pcie | 25.9× | derived |

---

## 4. Derived limits — all arithmetic shown

Capacities use **1 GB = 2^30 B** (the convention `nvidia-smi` follows: the A6000's
49140 MiB is 48 GiB). Bandwidths stay in decimal GB/s as measured. All matrices are fp32,
4 B per element.

### 4.1 Memory budgets

Free GPU memory on this server depends on who else is running, so we deliberately do
**not** derive any limit from observed free memory. Instead the implementation takes a
configurable budget and keeps 10 % headroom for workspace, alignment and fragmentation:

```
M_gpu = 0.9 x M_budget

M_budget = 24 GB  ->  M_gpu = 0.9 x 24 = 21.6 GB = 2.319282e10 B
M_budget = 48 GB  ->  M_gpu = 0.9 x 48 = 43.2 GB = 4.638565e10 B
```

The host is shared too — 1007 GiB total but only 705 GiB available at measurement time,
and 30 of 31 GiB of swap was already in use. We therefore budget 80 % of *available*
(not total) RAM:

```
M_host = 0.8 x 705 GiB = 564.0 GiB = 6.0559e11 B
```

This figure is a snapshot and will move as other users come and go.

### 4.2 Maximum n per strategy

**In-memory (full K, 4n² B):** n_max = sqrt(M_gpu / 4)

```
24 GB: sqrt(2.319282e10 / 4) = sqrt(5.798206e9) = 76,146.0  ->  n_max = 76,145
48 GB: sqrt(4.638565e10 / 4) = sqrt(1.159641e10) = 107,686.6 ->  n_max = 107,686
```

**S1 — symmetric tiles on the GPU (~2n² B):** n_max = sqrt(M_gpu / 2)

```
24 GB: sqrt(2.319282e10 / 2) = sqrt(1.159641e10) = 107,686.6 ->  n_max = 107,686
48 GB: sqrt(4.638565e10 / 2) = sqrt(2.319282e10) = 152,291.9 ->  n_max = 152,291
```

S1 buys exactly a sqrt(2) = 1.414× increase in n over the in-memory path — a useful but
bounded win, and the reason S1 alone does not solve the problem.

**S2 — upper-triangular tiles in pinned host memory (~2n² B):** n_max = sqrt(M_host / 2)

```
sqrt(6.0559e11 / 2) = sqrt(3.02795e11) = 550,268  ->  n_max = 550,268
```

S2 is host-bound, not GPU-bound: 550,268 / 152,291 = 3.6× over S1 at the 48 GB budget,
550,268 / 107,686 = 5.1× over the in-memory path at 48 GB, and 550,268 / 76,145 = 7.2×
over the in-memory path at 24 GB. The caveat is that this depends on the shared host's
available RAM.

**S3 — on-the-fly fused recomputation:** footprint is 4(nd + nk) B, linear in n, so
capacity is effectively never the binding constraint. At d = 50, k = 100 the capacity
limit would be

```
n <= M_gpu / (4(d+k)) = 2.319282e10 / 600 = 38,654,706   (24 GB budget)
n <= M_gpu / (4(d+k)) = 4.638565e10 / 600 = 77,309,411   (48 GB budget)
```

so S3 is **time-bound, not memory-bound**. Under a 60 s-per-iteration budget with
T_S3 = 2n²d / F_eff:

```
n = sqrt(T x F_eff / (2d))
  = sqrt(60 x 1.93536e13 / (2 x 50))
  = sqrt(1.161216e15 / 100)
  = sqrt(1.161216e13)
  = 3,407,662
```

At n = 3,407,662 the S3 footprint is 4 x 3,407,662 x (50 + 100) = 2.045e9 B = **1.90 GB**
— comfortably inside either budget, which confirms the limit is time, not capacity.

### 4.3 Max-n table

| Strategy | GPU bytes for K | n_max @ 24 GB | n_max @ 48 GB | Binding limit |
|---|---|---:|---:|---|
| In-memory (full K) | 4n² | 76,145 | 107,686 | GPU memory |
| S1 symmetric tiles | 2n² | 107,686 | 152,291 | GPU memory |
| S2 out-of-core | 2n² in host RAM | 550,268 | 550,268 | **host** memory (564 GiB) |
| S3 fused recompute | 0 (4(nd+nk) total) | 3,407,662 | 3,407,662 | **time** (60 s/iter, d = 50) |

S2 and S3 do not depend on the GPU budget, which is the entire point of the project.

### 4.4 Crossover dimensions

Setting the cost-model terms equal gives the d below which recomputing beats storing.

**S3 vs S2** (recompute beats streaming over PCIe) when d < F_eff / B_pcie:

```
d < 1.93536e13 / 2.625e10 = 737.3
```

**S3 vs in-memory** (recompute beats re-reading K from device DRAM) when
d < 2 x F_eff / B_gpu:

```
d < 2 x 1.93536e13 / 6.7935e11 = 3.87072e13 / 6.7935e11 = 56.98
```

Interpretation. Against PCIe the crossover is very high (d ~ 737): for essentially every
dataset Popcorn uses, recomputing K on the GPU is cheaper than shipping it across PCIe,
so S3 should dominate S2 whenever both are applicable, and S2's value is as the fallback
that preserves *exact* tile reuse when d is large. Against device DRAM the crossover is
d ~ 57, which lands right in the middle of real datasets (Acoustic d = 50, Letter d = 16
are below it; MNIST d = 780 and CIFAR-10 d = 3072 are far above). That is the interesting
regime the proposal targets: for low-dimensional data, Popcorn's store-and-reread design
is predicted to be *slower* than never storing K at all, independently of whether K fits.

### 4.5 Worked example

Take n = 2 × (in-memory n_max at the 48 GB budget) = 2 × 107,686.6 = **215,373**, with
d = 50. Full K would need 4n² = 1.855e11 B = 172.8 GB and the upper triangle 86.4 GB:
neither the in-memory path nor S1 can run this at all under a 48 GB budget.

```
T_S2 = 2n² / B_pcie   = 9.277129e10 / 2.625e10  = 3.5341 s  = 3534.1 ms/iteration
T_S3 = 2n² d / F_eff  = 4.638565e12 / 1.93536e13 = 0.2397 s  =  239.7 ms/iteration
```

S3 is predicted to be **14.7× faster** than S2 here, while needing only
4 × 215,373 × (50 + 100) = 1.29e8 B = **0.12 GB** on the GPU. With 30 iterations that is
106 s versus 7.2 s per clustering run. This is the headline prediction of the proposal.

### 4.6 Does Popcorn's largest dataset fit?

Acoustic, n = 78,823 (the largest dataset in the Popcorn paper):

```
K = 4n² = 4 x 78,823² = 4 x 6.21306e9 = 2.485226e10 B
       = 23.15 GB (2^30) = 24.85 GB (decimal)
```

| Budget | M_gpu | Full K = 23.15 GB | Verdict |
|---|---:|---|---|
| 24 GB | 21.6 GB | 23.15 > 21.6 | **Does not fit** |
| 48 GB | 43.2 GB | 23.15 < 43.2 | **Fits** |

So Acoustic is exactly the motivating case: Popcorn cannot run it on a 24 GB consumer GPU
under our budget policy, and even on the A6000 it consumes more than half the card. For
comparison, S1 would need only the upper triangle, 2n² = 11.57 GB, which fits under *both*
budgets; and S3 at d = 50, k = 100 needs 4n(d+k) = 4.73e7 B = **0.044 GB**, about 500×
less than the full K.

### 4.7 Reproducing this report

```bash
cd proposal/hwbench
make            # builds devinfo, bw_pcie, bw_gpu for sm_86
make run        # picks the least-loaded GPU and runs everything (a few seconds)
```

`run.sh` re-selects the GPU at launch, so results will differ with server load. Treat the
bandwidths above as the calibration for the cost model and re-measure before drawing
conclusions from new timings.
