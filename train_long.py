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
# os.environ['CUDA_VISIBLE_DEVICES']="7"

import numpy as np

from utils.timer import Timer

from torch.utils.data import DataLoader
from utils.loader_utils import FineSampler, EncoderBalancedSampler
import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, LikelihoodLoss, factor_loss
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d as o3d
from PIL import Image
from utils.visualize_utils import tensor2image
import cv2

from utils.debug_utils import check_anchor_modified
import utils.debug_utils

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)




def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')


def training(dataset, hyper, opt, pipe, frames_start_end, testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from, expname, args, load_iteration=None):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)

    gaussians = GaussianModel(hyper, opt, dataset.feat_dim, 10, dataset.voxel_size, dataset.update_depth,
                              dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist,
                              dataset.add_color_dist)

    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, frames_start_end, load_iteration=load_iteration)

    timer.start()

    scene_reconstruction(dataset, hyper, opt, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, opt.iterations, timer,
                         args, load_iteration=load_iteration)


from arguments.atgs_cfg import cfg


def optimizer_parameters(optimizer):
    if optimizer is None:
        return []
    return [param for group in optimizer.param_groups for param in group["params"]]


def optimizer_state_step(optimizer):
    """Recover the largest Adam step, or zero for a fresh optimizer."""
    if optimizer is None:
        return 0
    steps = []
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            steps.append(int(step.item()) if torch.is_tensor(step) else int(step))
    return max(steps, default=0)


def active_encoder_id(gaussians, timestamp):
    """Return the encoder index selected by the active spacetime field."""
    time_scalar = float(timestamp.item()) if torch.is_tensor(timestamp) else float(timestamp)
    return gaussians.dynamic_module.routing_encoder_id(time_scalar)


def sanitize_accumulated_gradients(optimizers):
    """Prevent one invalid micro-step from poisoning the whole accumulation window."""
    for optimizer in optimizers:
        for param in optimizer_parameters(optimizer):
            if param.grad is not None:
                param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


def average_accumulated_gradients(gaussians, micro_count, encoder_visit_counts, update_static):
    """Average shared gradients by batch size and routed encoder gradients by active visits."""
    if update_static:
        for param in optimizer_parameters(gaussians.optimizer):
            if param.grad is not None:
                param.grad.div_(micro_count)

    for name, param in gaussians.dynamic_module.named_parameters():
        if param.grad is None:
            continue
        divisor = micro_count
        if name.startswith("enc_models."):
            try:
                encoder_id = int(name.split(".", 2)[1])
                divisor = encoder_visit_counts.get(encoder_id, micro_count)
            except (ValueError, IndexError):
                pass
        param.grad.div_(max(divisor, 1))


def clip_accumulated_gradients(gaussians, max_norm, update_static):
    """Clip static, shared-dynamic, and each active encoder independently."""
    norms = {}
    if max_norm <= 0:
        return norms

    if update_static:
        static_params = [p for p in optimizer_parameters(gaussians.optimizer) if p.grad is not None]
        if static_params:
            norms["static"] = float(torch.nn.utils.clip_grad_norm_(static_params, max_norm))

    shared_params = []
    encoder_params = {}
    for name, param in gaussians.dynamic_module.named_parameters():
        if param.grad is None:
            continue
        if name.startswith("enc_models."):
            try:
                encoder_id = int(name.split(".", 2)[1])
                encoder_params.setdefault(encoder_id, []).append(param)
                continue
            except (ValueError, IndexError):
                pass
        shared_params.append(param)

    if shared_params:
        norms["dynamic_shared"] = float(torch.nn.utils.clip_grad_norm_(shared_params, max_norm))
    for encoder_id, params in encoder_params.items():
        norms[f"encoder_{encoder_id}"] = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
    return norms


def step_accumulated_gradients(gaussians, opt, micro_count, encoder_visit_counts, optimizer_update):
    # average the gradients: static are avaged by the micro_count, dynamic are avaged by the encoder_visit_counts
    average_accumulated_gradients(gaussians, micro_count, encoder_visit_counts, update_static=True)
    grad_norms = clip_accumulated_gradients(
        gaussians, float(opt.gradient_clip_norm), update_static=True
    )

    # warmup factor is used to warmup the learning rate
    warmup_updates = max(int(opt.lr_warmup_updates), 0)
    start_factor = min(max(float(opt.lr_warmup_start_factor), 0.0), 1.0)
    next_update = optimizer_update["count"] + 1
    if warmup_updates > 0:
        progress = min(next_update / warmup_updates, 1.0)
        warmup_factor = start_factor + (1.0 - start_factor) * progress
    else:
        warmup_factor = 1.0

    optimizers = [gaussians.optimizer, gaussians.dy_optimizer]

    saved_lrs = []
    for optimizer in optimizers:
        optimizer_lrs = []
        for group in optimizer.param_groups:
            lr = group["lr"]
            optimizer_lrs.append(lr)
            if any(param.grad is not None for param in group["params"]):
                group["lr"] = lr * warmup_factor
        saved_lrs.append(optimizer_lrs)

    for optimizer in optimizers:
        optimizer.step()

    # Warm-up is a transient multiplier; restore nominal scheduler/fixed rates.
    for optimizer, optimizer_lrs in zip(optimizers, saved_lrs):
        for group, lr in zip(optimizer.param_groups, optimizer_lrs):
            group["lr"] = lr

    gaussians.optimizer.zero_grad(set_to_none=True)
    gaussians.dy_optimizer.zero_grad(set_to_none=True)
    optimizer_update["count"] = next_update
    return grad_norms, warmup_factor


def save_launcher_shell_path(save_folder):
    import os

    bash_path = os.environ.get('bash_path', '')

    if bash_path == '':
        assert False, "Set bash_path in the environment before training."

    os.makedirs(save_folder, exist_ok=True)
    os.system(f'cp {bash_path} {save_folder}')

    return bash_path


def scene_reconstruction(dataset, hyper, opt, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter, timer, args, load_iteration=None):
    if load_iteration and checkpoint:
        assert False, "load_iteration and checkpoint cannot be used together"

    first_iter = 0

    setup_dir = os.path.join(args.model_path, "point_cloud", "iteration_" + str(load_iteration))
    gaussians.training_setup(opt, load_iteration, setup_dir)

    writer_path = os.path.join(scene.model_path, 'writer')
    os.makedirs(writer_path, exist_ok=True)
    writer = SummaryWriter(log_dir=writer_path)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if load_iteration:
        first_iter = load_iteration # For debug, useless

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, train_iter), desc="Training progress")
    first_iter += 1
    encoder_balanced_accumulation = True # bool(opt.encoder_balanced_accumulation and opt.dataloader)
    accumulation_steps = (
        max(int(cfg.levels), 1)
        if encoder_balanced_accumulation
        else max(int(opt.gradient_accumulation_steps), 1)
    )
    accumulated_micro_steps = 0
    encoder_visit_counts = {}
    optimizer_update = {
        "count": max(
            optimizer_state_step(gaussians.optimizer),
            optimizer_state_step(gaussians.dy_optimizer),
        ),
        "iteration": first_iter,
    }
    gaussians.optimizer.zero_grad(set_to_none=True)
    gaussians.dy_optimizer.zero_grad(set_to_none=True)
    print(
        f"[Gradient Accumulation] micro_steps={accumulation_steps}, "
        f"encoder_balanced={encoder_balanced_accumulation}, encoders={int(cfg.levels)}, "
        f"clip_norm={float(opt.gradient_clip_norm)}, "
        f"warmup_updates={int(opt.lr_warmup_updates)}, "
        f"optimizer_updates={optimizer_update['count']}"
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if not viewpoint_stack and not opt.dataloader:  # opt.dataloader True
        # dnerf's branch
        import copy
        train_cams = scene.getTrainCameras()
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if encoder_balanced_accumulation:
            sampler = EncoderBalancedSampler(viewpoint_stack, cfg.levels)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1, sampler=sampler, num_workers=16,
                                                collate_fn=list)
            random_loader = False
        elif opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1, sampler=sampler, num_workers=16,
                                                collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1, shuffle=True, num_workers=16,
                                                collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    load_in_memory = False
    total_frams = int(cfg.total_frames) if cfg.total_frames else 1400
    for iteration in range(first_iter, train_iter + 1):
        # network gui not available in scaffold-gs yet
        # if network_gui.conn == None:
        #     network_gui.try_connect()

        iter_start.record()
    
        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader.")
                if not random_loader and not encoder_balanced_accumulation:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size, shuffle=True,
                                                        num_workers=32, collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)
                viewpoint_cams = next(loader)
        else:
            idx = 0
            viewpoint_cams = []
            while idx < opt.batch_size:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue

        viewpoint_cam = viewpoint_cams[0]

        gaussians.update_learning_rate(iteration, opt, viewpoint_cam.time)

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, iteration=iteration,
                            visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        image, depth_map, viewspace_point_tensor, visibility_filter, offset_selection_mask, scaling, opacity, neural_points = \
            render_pkg["render"], render_pkg["depth_map"], render_pkg["viewspace_points"], render_pkg[
                "visibility_filter"], render_pkg["selection_mask"], \
                render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["neural_points"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image)[0])
        scaling_reg = scaling.prod(dim=1).mean()
        # PIL 2 img
        psnr_ = psnr(image, gt_image).mean().double()

        # for debug, visualize the image every 100 iterations
        if iteration % 100 == 0:
            os.makedirs(f"{scene.model_path}debug", exist_ok=True)
            torchvision.utils.save_image(image, f"{scene.model_path}debug/iteration_{iteration}.jpg")
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg

        if hyper.time_smoothness_weight != 0 and not opt.hash:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes,
                                                   hyper.plane_tv_weight, viewpoint_cam.time)
            loss += tv_loss

        primitive_type = cfg.primitive_type

        if primitive_type == '2dgs':
            lambda_normal = opt.lambda_normal if iteration > 2000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            loss += dist_loss
            loss += normal_loss

        del render_pkg

        if not torch.isfinite(loss).all():
            print(f"loss is not finite at iteration {iteration}; discarding accumulated gradients.")
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.dy_optimizer.zero_grad(set_to_none=True)
            accumulated_micro_steps = 0
            encoder_visit_counts.clear()
            writer.close()
            return

        loss.backward()
        accumulated_micro_steps += 1
        encoder_id = active_encoder_id(gaussians, viewpoint_cam.time)
        encoder_visit_counts[encoder_id] = encoder_visit_counts.get(encoder_id, 0) + 1
        sanitize_accumulated_gradients([gaussians.optimizer, gaussians.dy_optimizer])

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            total_point = gaussians.get_anchor.shape[0]
            if iteration % 10 == 0:
                writer.add_scalar("Loss", loss.item(), iteration)  # add scale

                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{total_point}"})
                progress_bar.update(10)


            if iteration == train_iter:
                progress_bar.close()

            statistics_active = iteration < opt.update_until and iteration > opt.start_stat
            if statistics_active:
                gaussians.training_statis(neural_points, viewspace_point_tensor, opacity, visibility_filter,
                                          offset_selection_mask, voxel_visible_mask)

            densification_due = (
                statistics_active
                and iteration >= opt.update_from
                and iteration % opt.update_interval == 0
            )
            encoder_batch_complete = (
                encoder_balanced_accumulation
                and len(encoder_visit_counts) == int(cfg.levels)
            )
            accumulation_complete = (
                encoder_batch_complete
                if encoder_balanced_accumulation
                else accumulated_micro_steps >= accumulation_steps
            )
            special_update_boundary = (
                iteration == train_iter
                or iteration in saving_iterations
                or iteration in testing_iterations
                or iteration in checkpoint_iterations
                or densification_due
            )
            force_update = accumulation_complete or special_update_boundary

            del viewspace_point_tensor, visibility_filter, offset_selection_mask
            del scaling, opacity, neural_points, voxel_visible_mask

            if force_update:
                optimizer_update["iteration"] = iteration
                # step the optimizer
                grad_norms, warmup_factor = step_accumulated_gradients(
                    gaussians,
                    opt,
                    accumulated_micro_steps,
                    encoder_visit_counts,
                    optimizer_update,
                )
                writer.add_scalar("Optimization/update", optimizer_update["count"], iteration)
                writer.add_scalar("Optimization/accumulated_micro_steps", accumulated_micro_steps, iteration)
                writer.add_scalar("Optimization/encoders_visited", len(encoder_visit_counts), iteration)
                writer.add_scalar("Optimization/warmup_factor", warmup_factor, iteration)
                for name, grad_norm in grad_norms.items():
                    writer.add_scalar(f"GradientNorm/{name}", grad_norm, iteration)
                accumulated_micro_steps = 0
                encoder_visit_counts.clear()

                # A save boundary can split a balanced batch. Restart the
                # sampler so the next update again begins with one sample per encoder.
                if (
                    encoder_balanced_accumulation
                    and special_update_boundary
                    and not encoder_batch_complete
                    and iteration < train_iter
                ):
                    loader = iter(viewpoint_stack_loader)
                # densification only be activated when the initial points is sparse
                if densification_due:
                    gaussians.adjust_anchor(iteration, check_interval=opt.update_interval,
                                            success_threshold=opt.success_threshold,
                                            grad_threshold=opt.densify_grad_threshold,
                                            min_opacity=opt.min_opacity)

            training_report(args, gaussians, tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, [pipe, background], scene.dataset_type,
                            logger)

            if iteration in saving_iterations:
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, dir_suffix="")

            if iteration == opt.update_until:
                for buffer_name in (
                    "opacity_accum",
                    "offset_gradient_accum",
                    "offset_denom",
                    "grad_max",
                    "grad_max_points",
                ):
                    if hasattr(gaussians, buffer_name):
                        delattr(gaussians, buffer_name)

            if iteration % 100 == 0:
                torch.cuda.empty_cache()

    writer.close()


def prepare_output_and_logger(expname):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        os.makedirs(args.model_path + "/logs/", exist_ok=True)
        tb_writer = SummaryWriter(args.model_path + "/logs/")
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

import concurrent.futures
def multithread_write(image_list, path, render_view_id, global_start_frames):
    os.makedirs(os.path.join(path, str(render_view_id)), exist_ok=True)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, str(render_view_id), '{0:05d}'.format(count) + ".png"))
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

def training_report(args, gaussians, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs, dataset_type, logger):
    # Report test and samples of training set
    import os
    render_feature = cfg.render_feature
    render_path = os.path.join(cfg.model_path, "test" + render_feature, f"ours_{iteration}", "renders")
    

    if iteration in testing_iterations:
        validation_configs = (
            {'name': 'test', 'cameras': [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras()))]},
        )
        global_start_frames = cfg.frame_start_idx
        global_end_frames = cfg.frame_end_idx

        per_frame_one_view = global_end_frames - global_start_frames
        
        render_view_ids = [0, 1, 2, 3]
        
        for render_view_id in render_view_ids:
            start_frame = render_view_id * per_frame_one_view
            end_frame = start_frame + per_frame_one_view
            batch_size = 60
            render_list = []

            batch_count = 0
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras'][start_frame:end_frame]):
                        voxel_visible_mask = prefilter_voxel(viewpoint, gaussians, renderArgs[0], renderArgs[1])
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, iteration=iteration,
                                                    visible_mask=voxel_visible_mask,
                                                    retain_grad=False, *renderArgs)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                        render_list.append(image)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        # mask=viewpoint.mask
                        psnr_ = psnr(image, gt_image).mean().double()
                        psnr_test += psnr_
                        if len(render_list) >= batch_size:
                            print(f"Writing batch {batch_count} for view {render_view_id}")
                            multithread_write(render_list, render_path, render_view_id, global_start_frames + idx - len(render_list) + 1)
                            
                            render_list = []
                            gt_list = []
                            batch_count += 1

                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])

                    logger.info(
                        "[ITER {} {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, render_view_id,config['name'], l1_test, psnr_test))
            
                    if len(render_list) > 0:
                        print(f"Writing final batch {batch_count} for view {render_view_id}")
                        # multithread_write(gt_list, gts_path, render_view_id, global_start_frames + (end_frame - start_frame) - len(render_list))
                        multithread_write(render_list, render_path, render_view_id, global_start_frames + (end_frame - start_frame) - len(render_list))

                    os.system(f"python image2video.py --model_path {cfg.model_path} --iteration {str(iteration)}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        t_start = time.time()

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize()
        t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]

        # error maps
        errormap = (rendering - gt).abs()

        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train=True, skip_test=False,
                wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                                  dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                                  dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist,
                                  dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count = render_set(dataset.model_path, "train", scene.loaded_iter,
                                                     scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps": train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter,
                                                    scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps": test_fps, })

    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

        if wandb is not None:
            wandb.log({"test_SSIMS": torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final": torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS": torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)

        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                             "PSNR": torch.tensor(psnrs).mean().item(),
                                             "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update(
            {"SSIM": {name: s for s, name in zip(torch.tensor(ssims).tolist(), image_names)},
             "PSNR": {name: p for p, name in zip(torch.tensor(psnrs).tolist(), image_names)},
             "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
             "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


def get_logger(path):
    import logging

    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

def save_run_codes(save_folder):
    import os
    from pathlib import Path

    os.makedirs(f'{save_folder}/code', exist_ok=True)

    include_patterns = [
        '*.py',
        'arguments/***',
        'gaussian_renderer/***',
        'scene/***',
        'utils/***',
    ]

    cur_folder = Path(__file__).resolve().parent

    cmd = f'rsync -av '
    
    cmd += f""

    for pattern in include_patterns:
        cmd += f"--include='{pattern}' "


    cmd += f"--exclude='*' {cur_folder}/ {save_folder}/code"
    
    os.system(cmd)
    print(f"Generated rsync command:\n{cmd}")


def load_env(args):
    from arguments.atgs_cfg import apply_atgs_cfg
    apply_atgs_cfg(args)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)  # 250 * 30 = 7500 7500
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 10880, 150000,200000, 250000, 300000, 350000, 500000, 700000, 1000000, 120_0000, 150_0000, 180_0000, 200_0000, 250_0000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[50000, 70000, 10000, 120000, 150000, 200000, 250000, 300000, 350000, 500000, 700000, 1000000, 120_0000, 150_0000, 180_0000, 200_0000, 250_0000])
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50000, 100000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default='-1')
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--restore_iteration", type=int, default=None)

    parser.add_argument("--frames_start_end", type=int, nargs=2, default=[0, 20], help="Start and end frames")

    args = parser.parse_args(sys.argv[1:])

    import shutil, os, sys
    print("PYTHON:", sys.executable)
    print("NINJA:", shutil.which("ninja"))
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"))

    # save_launcher_shell_path(args.model_path)
    # save_run_codes(args.model_path)

    if args.configs:
        import mmcv
        from utils.general_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    args.save_iterations.append(args.iterations)
    args.save_iterations = sorted(set(args.save_iterations))
    # enable logging

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f'args: {args}')

    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    load_env(args)

    from utils.history import init_encoder_visit_history

    init_encoder_visit_history()

    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.frames_start_end,
             args.test_iterations, args.save_iterations, \
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.model_path, args,
             load_iteration=args.restore_iteration)

