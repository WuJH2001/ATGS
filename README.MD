<h1 align="center"><strong>ATGS: Anchored Temporal Gaussian Splatting for Long Volumetric Video Representation</strong></h1>

<p align="center">
  <a href="https://wujh2001.github.io/">Jiahao Wu*</a><sup>1,2</sup>,
  <a hrdf="https://liangjie1999.github.io/liangjie.github.io/">Jie Liang*</a><sup>1,2</sup>,
  <a>Die Hu</a><sup>1</sup>,
  <a>Jiayu Yang</a><sup>2</sup>,
  <a>Kaiqiang Xiong</a><sup>1,2</sup>,
  <a>Xiaoyun Zheng</a><sup>2</sup>,
  <a>Xiang Li</a><sup>1,2</sup>,
  <a href="https://hans1984.github.io/">Chao Wang ✉</a><sup>2</sup>,
  <a href="https://www.ece.pku.edu.cn/info/1046/2147.htm">Ronggang Wang ✉</a><sup>1,2</sup>
</p>

<p align="center">
  <sup>1</sup> Peking University
  &nbsp;&nbsp;
  <sup>2</sup> Pengcheng Laboratory
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.22693-b31b1b.svg)](https://drive.google.com/file/d/1lRoi3-8B4lIjQSmGjS6nx0f9MLSQ5LGF/view?usp=sharing)
[![ProjectPage](https://img.shields.io/badge/%F0%9F%9A%80%20-Project%20Page-blue)](https://wujh2001.github.io/ATGS/)
[![Youtube](https://img.shields.io/badge/▶️-Youtube-red)]()

</div>

## Overview

**ATGS** represents long volumetric video with anchored temporal Gaussian splatting. A spacetime encoder (Hash or HexPlane) models dynamics over time clips and can be swapped by changing the config (`hash=True/False`).

## Installation

Follow the environment setup of Scaffold-GS / your existing CUDA PyTorch stack used by this repo (tinycudann, differentiable rasterizer, etc.). Point `PYTHON` in the shell scripts to your conda/venv interpreter if needed.

## Data

Prepare:

1. **Multi-view image sequence** under `-s` / `--source_path` (camera-major layout expected by the dataloader).
2. **Initial point cloud** via `--base_path` (e.g. `*.ply`). Frame range is controlled by `--frames_start_end START END`.

Example assets in this repo:

```text
data/360/250_points_enhanced.ply
```

## Training

Long-sequence training entry point: `train_long.py`. Experiment switches live in config files under `arguments/vrugz/`.

### Quick start (scripts)

Both encoders share the same training pipeline (`train_long.py`), time routing (`levels` / `clip_size`), and encoder-balanced gradient accumulation. Switch by config only:

| | **Hash encoder** | **HexPlane encoder** |
| --- | --- | --- |
| Config | `arguments/vrugz/basketball.py` (`hash=True`) | `arguments/vrugz/basketball_plane.py` (`hash=False`) |
| Backend | tinycudann HashGrid + MLP (`SpaceTimeHashingField`) | Multi-resolution spatiotemporal planes (`SpaceTimePlaneField`) |
| Speed | **Faster** forward / backward | Slower (plane grid lookup + interpolation) |
| GPU memory | **Higher** — each time clip keeps its own hash table | **Lower** — plane grids are usually more compact |
| Long sequences | e.g. 1400 frames, `clip_size=20` → 70 encoders; VRAM grows with `levels` | Same `levels` routing, but typically fits better on limited GPUs |
| Regularization | No plane TV loss | Optional plane smoothness / TV loss (`time_smoothness_weight`, etc.) |
| Checkpoint | `FDHash.pth` | `Planes.pth` |
| When to use | Default for speed; good when GPU memory is sufficient | Drop-in replacement when VRAM is tight or you want plane regularization |

**Hash encoder** (faster, higher GPU usage):

```bash
bash scripts/run_hash.sh
```

**HexPlane encoder** (drop-in replacement, lower memory):

```bash
bash scripts/run_plane.sh
```

To switch manually, only change `--configs` and keep other flags (`-s`, `-m`, `--frames_start_end`, `--base_path`) the same.

Edit paths inside the script before running:

| Variable / flag | Meaning |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | GPU id |
| `-s` | Source multi-view images |
| `-m` | Output / checkpoint directory |
| `--frames_start_end` | Inclusive-style frame window `[start, end)` used by configs |
| `--configs` | Hyperparameter file |
| `--base_path` | Initial PLY |

`bash_path` is exported by the script so the launcher can be copied next to the run for reproducibility.

### Manual command

```bash
python train_long.py \
  -s /path/to/images/ \
  -m output/exp_name/ \
  --frames_start_end 0 250 \
  --configs arguments/vrugz/basketball.py \
  --base_path /path/to/init_points.ply
```

Use `arguments/vrugz/basketball_plane.py` instead of `basketball.py` to train with planes (`hash=False`). Other training options (iterations, encoder levels, balanced accumulation, etc.) are set in those config files.

### Useful flags

- `--save_iterations` / `--test_iterations`: checkpoint and eval iterations
- `--restore_iteration N`: resume from `point_cloud/iteration_N`
- `--configs`: mmcv-style config merged into CLI args

Checkpoints are written to:

```text
<model_path>/point_cloud/iteration_<N>/
```

## Rendering

Entry point: `render.py`. Keep `-s`, `-m`, `--frames_start_end`, `--configs`, and `--base_path` consistent with the training run.

### Script

```bash
bash scripts/render.sh
```

Override via environment variables:

```bash
MODEL_PATH=output/360_2/0_250/ \
SOURCE_PATH=/path/to/images/ \
FRAMES_START=0 FRAMES_END=250 \
CONFIGS=arguments/vrugz/basketball.py \
BASE_PATH=/path/to/init_points.ply \
ITERATION=-1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/render.sh
```

- `ITERATION=-1`: load the latest checkpoint under `point_cloud/`
- Or set `ITERATION=100000` (example) for a specific iteration

### Manual command

```bash
python render.py \
  -s /path/to/images/ \
  -m output/exp_name/ \
  --frames_start_end 0 250 \
  --configs arguments/vrugz/basketball.py \
  --base_path /path/to/init_points.ply \
  --iteration -1 \
  --skip_train \
  --skip_video
```

Rendered images are saved under:

```text
<model_path>/testall/ours_<iteration>/renders/<view_id>/*.png
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{wu_atgs,
  author     = {Wu, Jiahao and Liang, Jie and Hu, Die and Yang, Jiayu and Xiong, Kaiqiang and Li, Xiang and Zheng, Xiaoyun and Wang, Chao and Wang, Ronggang},
  title      = {ATGS: Anchored Temporal Gaussian Splatting for Long Volumetric Video Representation},
  year       = {2026},
  issue_date = {July 2026},
  publisher  = {Association for Computing Machinery},
  address    = {New York, NY, USA},
  volume     = {45},
  number     = {4},
  issn       = {0730-0301},
  doi        = {10.1145/3811306},
  journal    = {ACM Trans. Graph.},
  articleno  = {110},
  numpages   = {13},
}
```

## Acknowledgement

This codebase builds upon LocalDyGS and related 3D Gaussian Splatting research. We thank the authors for open-sourcing their work.
