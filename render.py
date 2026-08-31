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
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import imageio
import numpy as np
import torch
from scene import Scene
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams ,ModelHiddenParams
from arguments.atgs_cfg import cfg
from gaussian_renderer import GaussianModel
from gaussian_renderer import prefilter_voxel, render
from time import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures
from PIL import Image



def multithread_write(image_list, path, render_view_id, global_start_frames):
    os.makedirs(os.path.join(path, str(render_view_id)), exist_ok=True)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, str(render_view_id), '{0:05d}'.format(count) + ".jpg"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, global_start_frames + index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
import matplotlib.pyplot as plt
import numpy as np

def draw_img(img, idx, title=""):
    # Flatten and convert tensor to numpy array
    ldr = img.detach().cpu().numpy().flatten()
    ldr = ldr[ldr>0.02]
    zero_array = np.zeros(140000)
    ldr = np.concatenate((ldr, zero_array))
    # Define bins for smaller and larger values
    bins = np.concatenate(([0, 1], np.linspace(1, ldr.max(), 49)))

    plt.figure(figsize=(10, 4))

    # Plot the histogram for the entire range
    counts, bins, patches = plt.hist(ldr, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    # If the maximum count is very large, plot separately for small and large values
    if counts.max() > 1000:
        plt.figure(figsize=(10, 4))
        counts, bins, patches = plt.hist(ldr[ldr < 100], bins=50, color='skyblue', edgecolor='black', alpha=0.7)

    # Annotate counts
    i = 0
    for count, bin_edge in zip(counts, bins[:-1]):
        if i == 1 or i ==21 or i == 31:
            if count > 0:
                plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, f'{int(count)}', 
                            ha='center', va='bottom', fontsize=10, color='black')
        i+=1

    # Set labels and title
    plt.title(title, fontsize=20)
    plt.xlabel('Dynamic Residual Feature Value', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set custom y-axis ticks
    plt.yticks(range(0, 150001, 30000))

    # Use scientific notation for y-axis
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.set_tick_params(pad=15)
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f"distribution_{idx}.jpg", format='jpg')
    plt.savefig(f"distribution_{idx}.pdf", format='pdf')
    plt.close()

def show_image(image,name):
    show_image = Image.fromarray(np.transpose(np.array(image.detach().clamp(0,1).cpu() * 255 ).astype(np.uint8), (1, 2, 0)))
    show_image.save(f'test_images/rgb_{name}.jpg')

def render_set_virtual(opt,model_path, name, iteration, views, gaussians, pipeline, background, cam_type, frames_start_end):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    render_view_ids = [0]

    global_start_frames = frames_start_end[0]
    global_end_frames = frames_start_end[1]
    for render_view_id in render_view_ids:
        render_images = []
        gt_list = []
        render_list = []
        # breakpoint()
        print("point nums:",gaussians._anchor.shape[0])
        all_time = 0
        per_view_frames = len(views)
        
        start_frame = render_view_id * per_view_frames

        gop_size = int(os.environ.get('gop_size', 20))
        per_gop_iter = int(os.environ.get('per_gop_iter', 20000))

        # end_frame = start_frame + min(per_view_frames, gop_size * (iteration // per_gop_iter))
        end_frame = (render_view_id + 1) * per_view_frames

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            gop_id = idx // gop_size

            print('gop_id', gop_id)

            time1 = time()
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, gop_id=0)
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            # rendering = render(view, gaussians, pipeline, background, render_anchor="all", visible_mask=voxel_visible_mask, retain_grad=retain_grad)["render"]
            rendering = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)["render"]

           
            time2 = time()
            all_time += (time2-time1)

            render_images.append(to8b(rendering).transpose(1,2,0))

            render_list.append(rendering)
            if name in ["train", "test"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                else:
                    gt  = view['image'].cuda()
                gt_list.append(gt)

        time2=time()
        print("FPS:",(end_frame - start_frame)/all_time)

        print("writing training images.") 
        multithread_write(gt_list, gts_path, render_view_id, global_start_frames) 
        print("writing rendering images.") 
        multithread_write(render_list, render_path, render_view_id, global_start_frames)

        os.makedirs(os.path.join(model_path, name, "ours_{}".format(iteration), 'videos', str(cfg.virtual_frame_interval), cfg.render_kind, str(render_view_id)), exist_ok=True) 
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'videos', str(cfg.virtual_frame_interval), cfg.render_kind, str(render_view_id), f'{global_start_frames}_{global_end_frames}.mp4'), render_images,fps=25)


def render_set(opt,model_path, name, iteration, views, gaussians, pipeline, background, cam_type, frames_start_end):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []

    # breakpoint()
    print("point nums:",gaussians._anchor.shape[0])
    all_time = 0

    per_frame_one_view = frames_start_end[1] - frames_start_end[0]

    # render_view_ids = [0, 1, 2, 3]
    render_view_ids = [0, 1]
    
    global_start_frames = frames_start_end[0]
    global_end_frames = frames_start_end[1]

    batch_size = 60

    for render_view_id in render_view_ids:
        start_frame = render_view_id * per_frame_one_view
        end_frame = start_frame + per_frame_one_view

        gt_list = []
        render_list = []
        batch_count = 0

        for idx, view in enumerate(tqdm(views[start_frame:end_frame], desc="Rendering progress")):
            time1 = time()
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            # rendering = render(view, gaussians, pipeline, background, render_anchor="all", visible_mask=voxel_visible_mask, retain_grad=retain_grad)["render"]
            rendering = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)["render"]
            time2 = time()
            all_time += (time2-time1)

            render_list.append(rendering)
            if name in ["train", "test"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                else:
                    gt  = view['image'].cuda()
                gt_list.append(gt)

            if len(render_list) >= batch_size:
                print(f"Writing batch {batch_count} for view {render_view_id}")
                multithread_write(gt_list, gts_path, render_view_id, global_start_frames + idx - len(render_list) + 1)
                multithread_write(render_list, render_path, render_view_id, global_start_frames + idx - len(render_list) + 1)
                
                render_list = []
                gt_list = []
                batch_count += 1

        if len(render_list) > 0:
            print(f"Writing final batch {batch_count} for view {render_view_id}")
            multithread_write(gt_list, gts_path, render_view_id, global_start_frames + (end_frame - start_frame) - len(render_list))
            multithread_write(render_list, render_path, render_view_id, global_start_frames + (end_frame - start_frame) - len(render_list))

def render_sets( opt , hyper, dataset : ModelParams, frames_start_end, iteration : int, pipeline : PipelineParams,  skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(hyper, opt,dataset.feat_dim, 10, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        # iteration = 7000
        scene = Scene(dataset, gaussians, frames_start_end = frames_start_end, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(opt,dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type, frames_start_end)
        if not skip_test:
            render_set(opt,dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type, frames_start_end)
        if not skip_video:
            render_set(opt,dataset.model_path,"video2",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type, frames_start_end)

        if not cfg.skip_blender:
            output_name = f'video_blender/{cfg.blender_name}'

            render_set_virtual(opt,dataset.model_path,output_name,scene.loaded_iter,scene.getBlenderCameras(), gaussians,pipeline,background,cam_type, frames_start_end=frames_start_end)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=[-1], type=int, nargs='+')
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--frames_start_end", type=int, nargs=2, default=[0, 300], help="Start and end frames")

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.general_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # enable logging
    # Initialize system state (RNG)
    safe_state(args.quiet)

    print(args.iteration)

    from arguments.atgs_cfg import apply_atgs_cfg
    apply_atgs_cfg(args)

    for i_iteration in args.iteration:
        render_sets(op.extract(args), hp.extract(args),   model.extract(args), args.frames_start_end, i_iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
    