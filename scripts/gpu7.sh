#!/usr/bin/env bash
# Sync the source to gpu7, build there with CUDA 12.4, pick the least-loaded
# GPU immediately before the run, and run a command on it.
#
#   scripts/gpu7.sh ./bin/test_kkm
#   scripts/gpu7.sh ./bin/kkm --strategy s2 --synthetic 200000 50 --k 10
#   GPU=7 scripts/gpu7.sh ...          # force a device
#
# Prints the load of every GPU and of the host right before the run, so the
# contention present during a measurement is on record (CLAUDE.md etiquette).
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=${GPU7_HOST:-gpu7}
REMOTE=${GPU7_DIR:-NYCU_PP-Final-Project}

rsync -a --delete --exclude .git --exclude .cuda --exclude build --exclude bin \
  --exclude results --exclude '.env*' ./ "$HOST:$REMOTE/"

# The command is passed as a single string, so quote arguments with spaces.
CMD=$(printf '%q ' "$@")
# .git is not synced; pass the commit so run reports (scripts/sweep.py) record it.
KKM_GIT=$(git describe --always --dirty 2>/dev/null || echo unknown)
ssh "$HOST" "GPU='${GPU:-}' KKM_GIT='$KKM_GIT' bash -s" <<EOF
set -euo pipefail
cd "$REMOTE"
export PATH=/usr/local/cuda-12.4/bin:\$PATH
make -s -j16 >&2
mkdir -p results
[ -n "${CMD// /}" ] || exit 0
if [ -z "\$GPU" ]; then
  # Lowest utilization first, then most free memory.
  GPU=\$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2,2n -k3,3nr | head -1 | cut -d, -f1 | tr -d ' ')
fi
echo "# \$(date -Is) host load: \$(cut -d' ' -f1-3 /proc/loadavg)  selected GPU \$GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader \
  | sed 's/^/# /'
CUDA_VISIBLE_DEVICES=\$GPU $CMD
EOF
