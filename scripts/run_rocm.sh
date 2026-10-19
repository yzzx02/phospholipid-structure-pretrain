#!/usr/bin/env bash
set -euo pipefail
project_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
rocm_python="${ROCM_PYTHON:-${HOME}/miniconda3/envs/torch-rocm721/bin/python}"
exec "$rocm_python" "$@"
