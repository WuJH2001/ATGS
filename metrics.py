#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, msssim
import lpipsPyTorch as lp
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import sys
from arguments import ModelParams
import numpy as np
import csv


def evaluate_memory_efficient_static(test_path, render_view_id=0, batch_size=8):
    """
    分块加载计算图像质量指标，避免OOM
    
    参数:
        test_path: 测试结果路径
        render_view_id: 渲染视角ID
        batch_size: 批处理大小，根据GPU内存调整
    """
    print(f"Selected test model is: {test_path}")

    gt_dir = os.path.join(test_path, "gt", str(render_view_id))
    
    renders_dir = os.path.join(test_path, "renders", str(render_view_id))
    
    image_files = sorted([f for f in os.listdir(renders_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    print(image_files)
    image_idxs = [int(item.split('.')[0]) for item in image_files]
    
    total_images = len(image_files)
    
    print(f"找到 {total_images} 张图像，使用批处理大小: {batch_size}")
    
    all_dssim1 = []
    all_dssim2 = []
    all_psnr = []
    all_lpips = []
    
    lpips_model = lp.LPIPS('vgg', '0.1').cuda()
    
    csv_path = os.path.join(test_path, f'{str(render_view_id)}.csv')
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["index", "psnr", "dssim1", "dssim2", "lpips"])
        
        for batch_start in tqdm(range(0, total_images, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_images)
            batch_files = image_files[batch_start:batch_end]
            
            renders_batch = []
            gts_batch = []
            
            for fname in batch_files:
                render_path = os.path.join(renders_dir, fname)
                gt_path = os.path.join(gt_dir, fname)
                
                render = Image.open(render_path)
                gt = Image.open(gt_path)
                target_size = gt.size

                render = render.resize(target_size, Image.Resampling.LANCZOS)
                
                render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
                gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
                
                renders_batch.append(render_tensor)
                gts_batch.append(gt_tensor)
            
            for idx, (render, gt) in enumerate(zip(renders_batch, gts_batch)):
                global_idx = batch_start + idx
                
                try:
                    dssim1 = (1 - ssim(render, gt)[0]) / 2
                    dssim2 = (1 - msssim(render, gt)) / 2
                    psnr_val = psnr(render, gt)
                    
                    lpips_val = lpips_model(render, gt).item()
                    
                    all_dssim1.append(dssim1.item())
                    all_dssim2.append(dssim2.item())
                    all_psnr.append(psnr_val.item())
                    all_lpips.append(lpips_val)
                    
                    writer.writerow([global_idx, psnr_val.item(), dssim1.item(), dssim2.item(), lpips_val])
                    
                except Exception as e:
                    print(f"计算图像 {global_idx} ({batch_files[idx]}) 时出错: {e}")
                    writer.writerow([global_idx, "ERROR", "ERROR", "ERROR", "ERROR"])
                
                finally:
                    del render, gt
                    torch.cuda.empty_cache()
            
            del renders_batch, gts_batch
            torch.cuda.empty_cache()
    
    avg_psnr = np.mean(all_psnr) if all_psnr else 0
    avg_dssim1 = np.mean(all_dssim1) if all_dssim1 else 0
    avg_dssim2 = np.mean(all_dssim2) if all_dssim2 else 0
    avg_lpips = np.mean(all_lpips) if all_lpips else 0
    
    print("\n===== 最终结果 =====")
    print(f"评估图像数量: {len(all_psnr)}/{total_images}")
    print(" Avg PSNR : {:>12.7f}".format(avg_psnr))
    print(" Avg DSSIM1 : {:>12.7f}".format(avg_dssim1))
    print(" Avg DSSIM2 : {:>12.7f}".format(avg_dssim2))
    print(" Avg LPIPS : {:>12.7f}".format(avg_lpips))
    
    with open(csv_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["avg", avg_psnr, avg_dssim1, avg_dssim2, avg_lpips])
    
    return {
        "psnr": avg_psnr,
        "dssim1": avg_dssim1,
        "dssim2": avg_dssim2,
        "lpips": avg_lpips
    }

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--input_dir_path", type=str, required=True)
    # example: ./output/360_2/0_250/testall/ours_100000/renders

    args = parser.parse_args(sys.argv[1:])

    render_view_ids = os.listdir(f"{args.input_dir_path}/renders")
    print(render_view_ids)

    for render_view_id in render_view_ids:
        evaluate_memory_efficient_static(args.input_dir_path, render_view_id, batch_size=100)
