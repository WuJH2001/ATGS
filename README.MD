<h1 align="center"><strong>ATGS: Anchored Temporal Gaussian Splatting for Long Volumetric Video Representation</strong></h1>

<p align="center">
  <a href="https://wujh2001.github.io/">Jiahao Wu*</a><sup>1,2</sup>,
  <a>Jie Liang*</a><sup>1,2</sup>,
  <a>Die Hu</a><sup>1</sup>,
  <a>Jiayu Yang</a><sup>2</sup>,
  <a>Kaiqiang Xiong</a><sup>1,2</sup>,
  <a>Xiaoyun Zheng</a><sup>2</sup>,
  <a>Xiang Li</a><sup>1,2</sup>,
  <a>Chao Wang ✉</a><sup>2</sup>,
  <a href="https://www.ece.pku.edu.cn/info/1046/2147.htm">Ronggang Wang ✉</a><sup>1,2</sup>
</p>

<p align="center">
  <sup>1</sup> Peking University
  <sup>2</sup> Pengcheng Laboratory
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.22693-b31b1b.svg)](https://dl.acm.org/doi/epdf/10.1145/3811306)
[![ProjectPage](https://img.shields.io/badge/%F0%9F%9A%80%20-Project%20Page-blue)](https://wujh2001.github.io/ATGS/)
[![Youtube](https://img.shields.io/badge/▶️-Youtube-red)](https://wujh2001.github.io/ATGS/static/videos/demo.mp4)
</div>


## 💡 Overview

**ATGS** represents long volumetric video with anchored temporal Gaussian splatting. A spacetime encoder (Hash or HexPlane) models dynamics over time clips and can be swapped by changing the config (`hash=True/False`).

## Installation


```bash
conda env create --file env.yml
```

After that, you need to install [tiny-cuda-nn (1.7)](https://github.com/NVlabs/tiny-cuda-nn) and [gsplat](https://github.com/nerfstudio-project/gsplat). 



## 🔧 Data Preparation

We use the VRU dataset as an example. First, download the VRU Long dataset (1400 frames) from [Hugging Face](https://huggingface.co/datasets/BestWJH/VRU_Basketball/tree/main). Follow the step-by-step instructions in [utils/multiview_data_process/readme.md](utils/multiview_data_process/readme.md) to process the data. Preprocessed cameras are available [here](https://drive.google.com/file/d/1AAThoE0g0_avz1sgoFhC9sqkMA_a0Qnv/view?usp=sharing).

You can also download our processed `ply` files from [Google Drive](https://drive.google.com/file/d/1FPEVP0iVmJShzisPlRbHSX9U04DWCXTN/view?usp=sharing) and place them under `data/`. or others.




## ⚡ Training

Long-sequence training entry point: `train_long.py`. Experiment switches live in config files under `arguments/`.

###  Quick start (scripts)

Both encoders share the same training pipeline (`train_long.py`), time routing (`levels` / `clip_size`), and encoder-balanced gradient accumulation. Switch by config only:

**Hash encoder** (faster, higher quality, higher GPU usage):

```bash
bash scripts/run_hash.sh
```

**HexPlane encoder** (drop-in replacement, lower memory):

```bash
bash scripts/run_plane.sh
```

To switch manually, only change `--configs` and keep other flags (`-s`, `-m`, `--frames_start_end`, `--base_path`) the same.

Edit paths inside the script before running:


| Variable / flag        | Meaning                                                     |
| ---------------------- | ----------------------------------------------------------- |
| `CUDA_VISIBLE_DEVICES` | GPU id                                                      |
| `-s`                   | Source multi-view images                                    |
| `-m`                   | Output / checkpoint directory                               |
| `--frames_start_end`   | Inclusive-style frame window `[start, end)` used by configs |
| `--configs`            | Hyperparameter file                                         |
| `--base_path`          | Initial PLY                                                 |


`bash_path` is exported by the script so the launcher can be copied next to the run for reproducibility.

### Manual command

```bash
python train_long.py \
  -s /path/to/vru_long/ \
  -m output/exp_name/ \
  --frames_start_end 0 250 \
  --configs arguments/vru/basketball.py \
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

## ⚡ Rendering

Entry point: `render.py`. Keep `-s`, `-m`, `--frames_start_end`, `--configs`, and `--base_path` consistent with the training run.

### Script

```bash
bash scripts/render.sh
```

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

Rendered images from `render.py` are saved under:

```text
<model_path>/test/ours_<iteration>/renders/<view_id>/*.jpg
```

Training-time evaluation in `train_long.py` uses `render_feature=all` and writes to `testall/ours_<iteration>/renders/<view_id>/*.png` instead.

## ⚡ Evaluation

Entry point: `metrics.py`. Point `--input_dir_path` to a render result folder that contains both `renders/` and `gt/` (same directory as above, without the trailing `renders/`).

### Script

```bash
bash scripts/metrics.sh
```

Edit `--input_dir_path` inside the script to match your render output (e.g. `test/ours_<iteration>` for `render.py`, or `testall/ours_<iteration>` for training eval).

### Manual command

```bash
python metrics.py \
  --input_dir_path output/exp_name/test/ours_<iteration>
```

It evaluates every view under `renders/`, writes per-image and average PSNR / DSSIM / LPIPS to `<input_dir_path>/<view_id>.csv`, and prints the averages.

## 🤗  Citation

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

## 🤗  Acknowledgement

This codebase builds upon LocalDyGS and related 3D Gaussian Splatting research. We thank the authors for open-sourcing their work.