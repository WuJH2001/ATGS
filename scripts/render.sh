#!/usr/bin/env bash
cd "$(dirname "$0")/.."


export CUDA_VISIBLE_DEVICES=2
python render.py \
  -s "/data2/dataset/longvideos/jpg/360_2/" \
  -m "output/360_2/0_250/" \
  --frames_start_end 0 250 \
  --configs "arguments/vrugz/basketball.py" \
  --base_path "./data/360/250_points_enhanced.ply" \
  --iteration -1 \
  --skip_train \
  --skip_video
