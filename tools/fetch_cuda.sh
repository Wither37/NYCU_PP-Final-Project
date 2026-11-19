#!/usr/bin/env bash
# Fetch a project-local CUDA 12.4 compiler (nvcc + cudart headers/static lib)
# into .cuda/, for machines that have no CUDA toolkit (e.g. a laptop or a VM
# without a GPU). gpu7 already has /usr/local/cuda-12.4 and does not need this.
#
# Nothing is installed system-wide; `rm -rf .cuda` undoes it. The archives are
# NVIDIA's official redistributables, checked against the sha256 in NVIDIA's
# manifest. Afterwards build with:  make NVCC=.cuda/bin/nvcc
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=https://developer.download.nvidia.com/compute/cuda/redist
MANIFEST=redistrib_12.4.1.json
DEST=.cuda
DL=$DEST/.download
mkdir -p "$DL"

curl -fsSL "$BASE/$MANIFEST" -o "$DL/$MANIFEST"

for pkg in cuda_nvcc cuda_cudart; do
  rel=$(jq -r --arg p "$pkg" '.[$p]["linux-x86_64"].relative_path' "$DL/$MANIFEST")
  sha=$(jq -r --arg p "$pkg" '.[$p]["linux-x86_64"].sha256' "$DL/$MANIFEST")
  file=$DL/$(basename "$rel")
  [ -f "$file" ] || curl -fsSL "$BASE/$rel" -o "$file"
  echo "$sha  $file" | sha256sum -c --quiet
  tar -xJf "$file" -C "$DEST" --strip-components=1
done
# nvcc.profile looks for lib64/; the redist archives ship lib/.
ln -sfn lib "$DEST/lib64"

"$DEST/bin/nvcc" --version | tail -2
