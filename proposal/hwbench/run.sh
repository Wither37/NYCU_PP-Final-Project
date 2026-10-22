#!/usr/bin/env bash
# Pick the least-loaded GPU on this shared server and run the microbenchmarks.
# Total runtime is a few seconds; peak allocation is 2 GiB device + 1 GiB pinned.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== nvidia-smi before ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader

# Lowest utilization first, then most free memory.
GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2,2n -k3,3nr | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$GPU
echo "=== selected GPU $GPU ==="

./devinfo
echo "=== B_pcie (pinned H2D/D2H, 1 GiB per copy, 5 trials) ==="
./bw_pcie 5
echo "=== B_gpu (read+write copy kernel, 2 GiB moved per trial, 5 trials) ==="
./bw_gpu 5

echo "=== nvidia-smi after ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i "$GPU"
