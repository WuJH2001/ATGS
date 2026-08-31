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

[![ACM TOG](https://img.shields.io/badge/ACM-TOG-blue)](https://dl.acm.org/doi/10.1145/3811306)
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

You can also download our processed `ply` files from [Google Drive](https://drive.google.com/file/d/1FPEVP0iVmJShzisPlRbHSX9U04DWCXTN/view?usp=sharing) and place them under `data/` (or your own path and pass it via `--base_path`).


## ⚡ Training

Long-sequence training entry point: `train_long.py`. Experiment switches live in config files under `arguments/vru/` or `arguments/vrugz/`.

**Hash encoder** (`hash=True`):

```bash
python train_long.py \
  -s /path/to/vru_long/ \
  -m output/exp_name/ \
  --frames_start_end 0 250 \
  --configs arguments/vru/basketball.py \
  --base_path /path/to/250_points_enhanced.ply
```

**HexPlane encoder** (`hash=False`): use `arguments/vru/basketball_plane.py` or `arguments/vrugz/basketball_plane.py` instead.

Other training options (iterations, encoder levels, balanced accumulation, etc.) are set in those config files.

| Flag / env | Meaning |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | GPU id (set in shell before running) |
| `-s` | Source multi-view images |
| `-m` | Output / checkpoint directory |
| `--frames_start_end` | Frame window `[start, end)` |
| `--configs` | Config file (mmcv-style) |
| `--base_path` | Initial PLY |
| `--restore_iteration N` | Resume from `point_cloud/iteration_N` |


## ⚡  Rendering

Entry point: `render.py`. Keep `-s`, `-m`, `--frames_start_end`, `--configs`, and `--base_path` consistent with the training run.

```bash
python render.py \
  -s /path/to/images/ \
  -m output/exp_name/ \
  --frames_start_end 0 250 \
  --configs arguments/vru/basketball.py \
  --base_path /path/to/init_points.ply \
  --iteration -1 \
  --skip_train \
  --skip_video
```

Outputs are written to `<model_path>/test/ours_<iteration>/renders/<view_id>/*.jpg` (and matching `gt/`).


## ⚡  Evaluation

Entry point: `metrics.py`. Set `--input_dir_path` to the folder that contains both `renders/` and `gt/` (not the `renders/` folder itself), e.g. `output/exp_name/test/ours_<iteration>` for `render.py`, or `testall/ours_<iteration>` for training-time eval.

```bash
python metrics.py \
  --input_dir_path output/exp_name/test/ours_<iteration>
```

It evaluates every view under `renders/`, writes per-image and average PSNR / DSSIM / LPIPS to `<input_dir_path>/<view_id>.csv`, and prints the averages.

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