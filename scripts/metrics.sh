#!/usr/bin/env bash
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=2

python metrics.py \
  --input_dir_path "output/360_2/0_250/testall/ours_100000"
