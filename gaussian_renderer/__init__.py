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

import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
import tinycudann as tcnn
import torch.nn.functional as F

import os
import open3d as o3d
import torch

import numpy as np

import gsplat

from gsplat.cuda._wrapper import fully_fused_projection_2dgs
from arguments.atgs_cfg import cfg


def save_anchor(anchor):
    anchor_1 = anchor.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(anchor_1)

    o3d.io.write_point_cloud("anchor.ply", pcd)
    print("保存完成 ✅ anchor.ply")

# time_mask
def expand_visible_indices(time_mask, visible_mask, constant):
    """
    从布尔掩码中提取 True 的索引，并扩展为两列：[index, constant]

    Args:
        visible_mask (np.ndarray): 1D 布尔数组
        constant (int or float): 要附加的常量值

    Returns:
        np.ndarray: shape (K, 2), 每行是 [original_index, constant]
    """

    mask_np = time_mask.detach().cpu().numpy()

    indices = np.flatnonzero(mask_np)

    expanded = np.column_stack([indices, np.full_like(indices, constant)])

    return expanded[visible_mask.cpu().numpy()]

def get_time_mask_by_window(pc, time):
    timestamp = torch.tensor(time)
    frame_interval = int(cfg.frame_interval)
    window_size = int(cfg.time_window_size)

    num_frames = int(cfg.total_point_frames) * frame_interval
    start_frame_idx = int(cfg.frame_start_idx)

    center_idx = int(timestamp * (num_frames - 1)) + start_frame_idx
    half_window = window_size // 2

    start_idx = max(0, center_idx - half_window)
    end_idx = min(num_frames + start_frame_idx, center_idx + half_window + 1)

    time_window_slice = pc.point_times_list[:, (start_idx // frame_interval):(
            end_idx // frame_interval)]
    time_mask = (time_window_slice.max(dim=1)[0] > 0)

    # Keep frame-0 points always active
    frame_0_mask = pc.point_times_list[:, 0] > 0
    time_mask = torch.logical_or(time_mask, frame_0_mask)

    return time_mask


def syncTime(pc, timestamp, cam):
    if bool(cfg.open_sync_time):
        syncDict = pc.syncDict
        image_path = cam.image_path
        frame_idx = int(image_path.split('/')[-1].split('cam')[-1].split('.png')[0])

        return timestamp - syncDict[frame_idx]
    else:
        return timestamp


def generate_full_neural_gaussians(viewpoint_camera, pc: GaussianModel, iteration=None, visible_mask=None,
                                   is_training=False, timestamp=None, opt_thro=0.0):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    if timestamp == None:
        timestamp = torch.tensor(viewpoint_camera.time)

    old_timestamp = timestamp.clone()
    timestamp = syncTime(pc, timestamp, viewpoint_camera)
    
    time_mask = get_time_mask_by_window(pc, viewpoint_camera.time)

    anchor = pc.get_anchor[time_mask][visible_mask]  # [N,3]

    if anchor.shape[0] == 0:
        print('error')

    timestamp = timestamp.to(anchor.device).repeat(anchor.shape[0], 1)
    old_timestamp = old_timestamp.to(anchor.device).repeat(anchor.shape[0], 1)
    voxel_feat = pc.voxel_grid.query_features(anchor)


    dy_feat, dy_factor = pc.dynamic_module(anchor, timestamp, timestamp)

    sta_feat = pc._anchor_feat[time_mask][visible_mask]  # [N,32]

    grid_scaling_original = pc.get_scaling[time_mask][visible_mask]  # [N,6]

    sta_feat = sta_feat + voxel_feat
    feat = torch.cat([sta_feat, dy_feat], dim=-1)

    ob_view = anchor - viewpoint_camera.camera_center.cuda().unsqueeze(0)
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    num_points = feat.shape[0]
    neural_opacity = pc.get_opacity_mlp(feat)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity > opt_thro)
    mask = mask.view(-1)

    # select opacity
    opacity = neural_opacity[mask]

    color = pc.get_color_mlp(feat)
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [mask]

    scale_rot = pc.get_cov_mlp(feat)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [mask]

    offsets = pc.get_offset_mlp(feat)
    offsets = offsets.view([-1, 3])  # pc._offset[pc.dynamic_mask]  #  [N,10,3]
    del feat
    grid_scaling = grid_scaling_original

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    del concatenated
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    del concatenated_repeated
    masked = concatenated_all[mask]
    del concatenated_all
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    del masked

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])  # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:, 3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    if scaling.shape[0] == 0:
        pass

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask,
    else:
        return xyz, color, opacity, scaling, rot


def save_ply(points, name):
    import open3d as o3d
    points_np = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    o3d.io.write_point_cloud(f"{name}.ply", pcd)

    print(f"保存完成: {name}.ply")

def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, opt_thro = 0.0, scaling_modifier = 1.0, iteration = 30000 , retain_grad=False , render_anchor=False, visible_mask=None):
    primitive_type = cfg.primitive_type

    if primitive_type == '3dgs':
        return render_3dgs(viewpoint_camera, pc, pipe, bg_color, opt_thro, scaling_modifier, iteration, retain_grad, render_anchor, visible_mask)
    elif primitive_type == '2dgs':
        return render_2dgs(viewpoint_camera, pc, pipe, bg_color, opt_thro, scaling_modifier, iteration, retain_grad, render_anchor, visible_mask)
    else:
        assert False, f"Current codes don't support primivie_tpye: {primitive_type} "

def prefilter_voxel(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, gop_id=None, timestamp=None):
    primitive_type = cfg.primitive_type

    if primitive_type == '3dgs':
        return prefilter_voxel_3dgs(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    elif primitive_type == '2dgs':
        return prefilter_voxel_2dgs(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    else:
        assert False, f"Current codes don't support primivie_type: {primitive_type} "


def render_3dgs(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, opt_thro=0.0,
           scaling_modifier=1.0, iteration=30000, retain_grad=False, render_anchor=False, visible_mask=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    # filter out the gaussians that are not visible
    # if iteration >= 10000 and iteration <= 20000:
    #     opt_thro = (iteration / 1000) * 0.001 - 0.01
    # elif iteration > 20000:
    #     opt_thro = 0.01

    opt_thro = 0.0

    xyz, color, opacity, scaling, rot, neural_opacity, mask = \
        generate_full_neural_gaussians(viewpoint_camera, pc, iteration=iteration, visible_mask=visible_mask,
                                        is_training=is_training, opt_thro=opt_thro)

    if iteration % 1001 == 0:
        print(f"Temporal Gaussian points number: {xyz.shape[0]}, total seeds: {pc.get_anchor.shape[0]}")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
            # dy_dynamics.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_sta_image = None
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "render_sta_image": rendered_sta_image,
                "depth_map": None,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "neural_points": xyz
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                }


def render_2dgs(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, opt_thro = 0.0, scaling_modifier = 1.0, iteration = 30000 , retain_grad=False , render_anchor=False, visible_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    opt_thro = 0.0
    xyz, color, opacity, scaling, rot, neural_opacity, mask = \
        generate_full_neural_gaussians(viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training, opt_thro=opt_thro)
        

    if iteration % 1000 == 0 and iteration % 5000 != 0:
        print(f"Temporal Gaussian points number: {xyz.shape[0]}, total seeds: {pc.get_anchor.shape[0]}")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
            # dy_dynamics.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) 
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    # print('near_plane: ', float(os.environ.get('near_plane', 0.01)))
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)

    render_colors, \
    render_alphas, \
    render_normals, \
    render_normals_from_depth, \
    render_distort, \
    render_median, \
    info = \
    gsplat.rasterization_2dgs(
        means=xyz,  # [N, 3] 
        quats=rot,  # [N, 4] 
        scales=scaling,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=color,
        viewmats=viewmat[None].cuda(),  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]  
        backgrounds=bg_color[None],   #backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=None,
        render_mode="RGB+ED",
        near_plane=float(cfg.near_plane),
        depth_mode='expected',
        absgrad=bool(cfg.absgrad)
    )

    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    radii, _ = torch.max(radii, dim=1)
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "render_sta_image": None,
                "depth_map":  None,
                "viewspace_points": info["means2d"],
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "neural_points":xyz,
                'rend_alpha': render_alphas,
                'rend_normal': render_normals,
                'rend_dist': render_distort,
                'surf_depth': depth,
                'surf_normal': render_normals_from_depth,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }
    

def prefilter_voxel_3dgs(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    time_mask = get_time_mask_by_window(pc, viewpoint_camera.time)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # false
        cov3D_precomp = pc.get_covariance(scaling_modifier)[time_mask]
    else:
        scales = pc.get_scaling[time_mask]
        rotations = pc.get_rotation[time_mask]  # [N,4]

    radii_pure = rasterizer.visible_filter(means3D=means3D[time_mask],
                                           scales=scales[:, :3],
                                           rotations=rotations,
                                           cov3D_precomp=cov3D_precomp)

    return radii_pure > 0

def prefilter_voxel_2dgs(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    time_mask = get_time_mask_by_window(pc, viewpoint_camera.time)

    means = pc.get_anchor[time_mask]
    scales = pc.get_scaling[:, :3][time_mask]
    quats = pc.get_rotation[time_mask]
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    Ks = torch.tensor([
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],device="cuda",)[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda()

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    densifications = (
        torch.zeros((C, N, 2), dtype=means.dtype, device="cuda")
    )
    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        int(viewpoint_camera.image_width),
        int(viewpoint_camera.image_height),
        0.3, # eps2d=0.3
        0.01,
        1e10, # far_plane=1e10
        0.0, # radius_clip=0.0
        False, # packed=False
        False, # sparse_grad=False
    )
    
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None

    mask1 = radii.squeeze(0) > 0
    mask2 = radii.squeeze(0) < 1600
    
    # print(torch.logical_and(mask1, mask2).all(dim=1).shape)
    return torch.logical_and(mask1, mask2).all(dim=1)