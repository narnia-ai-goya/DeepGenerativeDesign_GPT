#!/usr/bin/env bash
# Run the image-only chair study from a shell that exposes /dev/nvidia*.
set -euo pipefail

PROJECT_ROOT=/home/goya/SDL/3d_qd
PYTHON_BIN=/home/goya/miniconda3/envs/direct3ds2/bin/python
GPU_INDEX=${1:-4}
EXPERIMENT_ROOT="$PROJECT_ROOT/experiments/chair/sofa_style_2026-09-28/image_only_bc_2026-10-02/complex_truss_armchair"

if [[ ! -e /dev/nvidiactl || ! -e /dev/nvidia${GPU_INDEX} ]]; then
  echo "GPU device nodes are unavailable in this shell: /dev/nvidiactl or /dev/nvidia${GPU_INDEX}" >&2
  exit 2
fi
if ! nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,memory.free --format=csv,noheader; then
  echo "NVIDIA driver is not accessible from this shell." >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/mpl_chair_2026_10_02
mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" codebase/run_chair_image_only_2026_10_02.py dense --gpu "$GPU_INDEX"
test -s "$EXPERIMENT_ROOT/dense_pw2/mesh_dense.obj"
test -s "$EXPERIMENT_ROOT/dense_pw2_cache.npz"
"$PYTHON_BIN" codebase/run_chair_image_only_2026_10_02.py sparse --gpu "$GPU_INDEX"
test -s "$EXPERIMENT_ROOT/sparse_pw2_d13/generation/mesh.obj"

echo "Dense: $EXPERIMENT_ROOT/dense_pw2/mesh_dense.obj"
echo "Sparse: $EXPERIMENT_ROOT/sparse_pw2_d13/generation/mesh.obj"
