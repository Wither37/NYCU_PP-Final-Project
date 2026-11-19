#!/usr/bin/env bash
# Build locally (no GPU needed), copy the binaries to the Windows test PC's WSL
# and run a command there. Interim test target until gpu7 access is set up.
#
#   scripts/pc.sh ./bin/test_kkm
#   scripts/pc.sh ./bin/kkm --strategy s2 --synthetic 20000 50 --k 10
#
# Assumes `ssh pc` reaches the PC and WSL distro Ubuntu-24.04 (user dev).
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=/home/dev/workspace/personal/NYCU/NYCU_PP-Final-Project
NVCC=${NVCC:-.cuda/bin/nvcc}
[ -x "$NVCC" ] || NVCC=nvcc

make -s -j"$(nproc)" NVCC="$NVCC" ARCH="86 89" >&2
ssh pc "wsl -d Ubuntu-24.04 -- mkdir -p $REMOTE/bin $REMOTE/results"
rsync -a --rsync-path="wsl rsync" bin/ "pc:$REMOTE/bin/"
[ $# -gt 0 ] || exit 0
ssh pc "wsl -d Ubuntu-24.04 --cd $REMOTE -- $*"
