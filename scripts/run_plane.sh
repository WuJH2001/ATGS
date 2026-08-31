#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/data1/wjh/env/scaffold/bin/python"
MAIN_SCRIPT="${ROOT_DIR}/train_long.py"

export CUDA_VISIBLE_DEVICES=2
export bash_path="$(realpath "${BASH_SOURCE[0]}")"

cd "${ROOT_DIR}"

ARGS=(
    -s "/data2/dataset/longvideos/jpg/360_2/"
    -m "output/360_2/0_250/"
    --frames_start_end 0 250
    --configs "arguments/vrugz/basketball_plane.py"
    --base_path "/data2/liangjie/dynamic/ATGS/data/360/250_points_enhanced.ply"
)

echo ">>> GPU: ${CUDA_VISIBLE_DEVICES}"
echo ">>> Running train_long.py for frames [0, 250) ..."
"${PYTHON}" "${MAIN_SCRIPT}" "${ARGS[@]}"
status=$?
echo "Python exit code: ${status}"
exit "${status}"
