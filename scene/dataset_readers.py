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
import glob
import sys
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple, Optional
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_intrAextr_npz, read_extrinsics_text_list
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from arguments.atgs_cfg import cfg
import torchvision.transforms as transforms
import torch
import torch.nn as nn

from scene.get_render_poses import get_render_poses,interpolation_frames

from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int
    blender_cameras: Optional[list]

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # print(f'FovX: {FovX}, FovY: {FovY}')

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # print(f'image: {image.size}')

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


from collections import defaultdict

def build_voxel_index(positions: np.ndarray, voxel_size: float):
    """
    positions: (N,3)
    voxel_size: voxel edge length (world coordinates)
    return:
      voxel_dict: dict[(ix,iy,iz)] -> list of global point indices
      origin: positions.min(axis=0), used to shift coordinates into a positive range (more stable)
    """
    origin = positions.min(axis=0)
    ijk = np.floor((positions - origin) / voxel_size).astype(np.int32)
    voxel_dict = defaultdict(list)
    for idx, key in enumerate(map(tuple, ijk)):
        voxel_dict[key].append(idx)
    return voxel_dict, origin

def query_nearest_in_frame_with_voxels(
    positions: np.ndarray,
    time_list_bool: np.ndarray,   # (N,T) bool
    voxel_dict,
    origin: np.ndarray,
    voxel_size: float,
    frame_id: int,
    target_xyz,
    ring: int = 1,
    max_expand: int = 4
):
    """
    Among points visible at frame_id, find the nearest point to target_xyz (global index).
    Candidates come from voxel buckets near the target voxel to reduce search cost.
    """
    target = np.asarray(target_xyz, dtype=np.float32)
    T = time_list_bool.shape[1]
    if frame_id < 0 or frame_id >= T:
        raise IndexError(f"frame_id {frame_id} out of range [0,{T-1}]")

    base = np.floor((target - origin) / voxel_size).astype(np.int32)

    def gather_candidates(r):
        cand = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    key = (int(base[0] + dx), int(base[1] + dy), int(base[2] + dz))
                    if key in voxel_dict:
                        cand.extend(voxel_dict[key])
        return np.array(cand, dtype=np.int64)

    cur = ring
    candidates = gather_candidates(cur)
    while candidates.size == 0 and cur < max_expand:
        cur += 1
        candidates = gather_candidates(cur)

    if candidates.size == 0:
        return None

    vis = time_list_bool[candidates, frame_id]
    candidates = candidates[vis]
    if candidates.size == 0:
        cur2 = cur
        while candidates.size == 0 and cur2 < max_expand:
            cur2 += 1
            cand2 = gather_candidates(cur2)
            if cand2.size == 0:
                continue
            vis2 = time_list_bool[cand2, frame_id]
            candidates = cand2[vis2]

        if candidates.size == 0:
            return None

    pts = positions[candidates]
    dist2 = np.sum((pts - target[None, :]) ** 2, axis=1)
    j = int(np.argmin(dist2))
    gid = int(candidates[j])
    return gid, float(np.sqrt(dist2[j])), cur

def plot_appearance_histogram(time_list, max_show=10, save_path='/data2/liangjie/dynamic/longvideo_odd_dynamic/vis/freq_of_point.png'):
    import matplotlib.pyplot as plt
    """
    Plot a histogram of how often each point appears in time_list.

    Args:
        time_list (np.ndarray): shape (N, T); 1 means the point appears in that frame.
        max_show (int): maximum count shown on the x-axis (e.g. only show distribution up to 10)
        save_path (str or None): if provided, save the figure (e.g. 'appearance_hist.png')
    """
    appearance_counts = time_list.sum(axis=1)  # shape (N,)

    print(f'Total appearances (including singletons): {appearance_counts.sum(axis=0)}')
    once_points_num = np.sum(appearance_counts == 1)
    print(f'Total appearances (excluding singletons): {appearance_counts.sum(axis=0) - once_points_num}')

    counts = appearance_counts[appearance_counts <= max_show]
    over_max = np.sum(appearance_counts > max_show)

    plt.figure(figsize=(10, 6))
    bins = np.arange(0.5, max_show + 1.5, 1)
    n, bins_edges, patches = plt.hist(counts, bins=bins, rwidth=0.8, color='skyblue', edgecolor='black', alpha=0.7)

    for i in range(len(n)):
        plt.text(bins_edges[i] + 0.5, n[i] + max(n)*0.01, str(int(n[i])), ha='center', va='bottom')

    if over_max > 0:
        plt.text(max_show + 0.5, max(n)*0.9, f'>{max_show}: {over_max}', 
                 ha='center', va='top', color='red', fontsize=10)

    plt.title('Distribution of Point Appearance Counts Across Frames')
    plt.xlabel('Number of frames a point appears in')
    plt.ylabel('Number of points')
    plt.xticks(range(1, max_show + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    if save_path:
        dir = os.path.dirname(save_path)
        os.makedirs(dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()

def fetchPly(path):
    """Load point cloud from PLY; read per-point time list from {ply}_time_list.npy, or fall back to PLY time_list."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])

    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])

    time_list_path = f"{path.replace('.ply', '')}_time_list.npy"
    print(f"[INFO] Loading time information from external file: {time_list_path}")

    if os.path.exists(time_list_path):
        point_times_list = np.load(time_list_path)
        print(f"[INFO] Loaded time data shape: {point_times_list.shape}, dtype: {point_times_list.dtype}")
        if point_times_list.shape[0] != positions.shape[0]:
            print(f"[ERROR] Time data point count ({point_times_list.shape[0]}) does not match point cloud count ({positions.shape[0]})!")
            raise ValueError("Time data and point cloud have mismatched point counts")
    else:
        print(f"[WARNING] Time file not found: {time_list_path}; reading time_list from PLY")
        point_times_list = np.array(vertices['time_list'])

    def verify_time_points(positions, colors, time_list, output_dir="./output/time_verification", num_samples=5):
        """
        Randomly sample a few time frames and save the corresponding point clouds as PLY for visual comparison.
        """
        import random

        if time_list is None:
            print("Skipped: no time list available")
            return

        os.makedirs(output_dir, exist_ok=True)

        if time_list.dtype != bool:
            time_list_bool = time_list > 0
        else:
            time_list_bool = time_list

        num_frames = time_list_bool.shape[1]

        points_per_frame = time_list_bool.sum(axis=0)
        valid_frames = np.where(points_per_frame > 0)[0]

        print(f"\nVerifying time correspondence")

        selected_frames = random.sample(valid_frames.tolist(), min(num_samples, len(valid_frames)))
        selected_frames.sort()
        # selected_frames = [0]

        for frame_id in selected_frames:
            mask = time_list_bool[:, frame_id]
            frame_points = positions[mask]
            frame_colors = colors[mask]

            print(f"Frame {frame_id:03d}: {frame_points.shape[0]} points")

            output_path = os.path.join(output_dir, f"frame_{frame_id:03d}_points.ply")
            save_simple_ply(output_path, frame_points, frame_colors)
            print(f"Saved: {output_path}")


    def save_simple_ply(path, xyz, rgb):
        """Save a simple PLY file (position + color)."""
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))

        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(path)

    if bool(os.environ.get('VERIFY_TIME_CORRESPONDENCE', False)):
        verify_time_points(positions, colors, point_times_list,
                           output_dir=os.environ.get('VERIFY_OUTPUT_DIR', './output/time_verification'),
                           num_samples=int(os.environ.get('VERIFY_NUM_SAMPLES', 5)))
        print("Finish verify_time_points")

    if bool(os.environ.get("DUMP_NEAREST_VOXEL", False)):
        tl = point_times_list
        if tl.dtype == np.bool_:
            tl = tl
        elif tl.dtype == object:
            tl = np.stack([np.asarray(t, dtype=bool) for t in tl], axis=0)
        else:
            tl = tl.astype(bool)

        voxel_size = float(os.environ.get("VOXEL_SIZE", "0.01"))
        voxel_dict, origin = build_voxel_index(positions, voxel_size)

        queries = [
            ("frame_000", 0, (1.081067, 0.642313, 1.040993)),
            ("frame_113", 113, (0.891131, 0.831370, 1.054423)),
            ("frame_108", 108, (0.128795, 0.750361, 1.202024)),
            ("frame_097", 97, (1.192728, 0.869199, 0.846768)),
            ("frame_053", 53, (1.000685, 0.861241, 0.986787)),
            ("frame_049", 49, (0.229282, 0.965476, 0.657244)),
        ]

        out_path = os.environ.get("NEAREST_OUT", "./output/indices_voxel.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for name, fid, xyz in queries:
                ret = query_nearest_in_frame_with_voxels(
                    positions=positions,
                    time_list_bool=tl,
                    voxel_dict=voxel_dict,
                    origin=origin,
                    voxel_size=voxel_size,
                    frame_id=fid,
                    target_xyz=xyz,
                    ring=int(os.environ.get("VOXEL_RING", "1")),
                    max_expand=int(os.environ.get("VOXEL_MAX_EXPAND", "4")),
                )
                if ret is None:
                    line = f"[MISS] {name} frame={fid} target={xyz} -> no candidates\n"
                else:
                    gid, dist, used_ring = ret
                    px, py, pz = positions[gid]
                    line = (f"[OK] {name} frame={fid} target={xyz} -> gid={gid} "
                            f"nearest=({px:.6f},{py:.6f},{pz:.6f}) dist={dist:.6f} ring={used_ring}\n")
                print(line.strip())
                f.write(line)

        print(f"[INFO] wrote: {out_path}")



    # plot_appearance_histogram(point_times_list)
    return BasicPointCloud(
        points=positions,
        colors=colors,
        normals=normals,
        point_times=None,
        point_times_list=point_times_list,
        point_times_bit_list=None,
    )

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    return ply_data

def colmap_format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[idx//dataset.N_frames][0], image.shape[1])
            FovY = focal2fov(dataset.focal[idx//dataset.N_frames][1], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))


    return cameras



def readCameraFromBlender(path):
    cameras_extrinsic_file = os.path.join(path, "images.txt")
    cameras_intrinsic_file = os.path.join(path, "cameras.txt")
    cam_extrinsics = read_extrinsics_text_list(cameras_extrinsic_file)  #
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    return cam_extrinsics, cam_intrinsics

def readColmapSceneInfo(path, images, eval, lod, frames_start_end=[0,300], llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    from scene.colmap_dataset import Colmap_Dataset, Colmap_Dataset_from_Blender

    test_view_ids = list(range(0, len(cam_extrinsics), llffhold))
    print('llffhold:', llffhold, 'test_view_ids:', test_view_ids)

    train_num = [i for i in range(len(cam_extrinsics)) if i not in test_view_ids]
    train_exr = [cam_extrinsics[i] for i in range(len(cam_extrinsics)) if i not in test_view_ids]
    test_exr = [cam_extrinsics[i] for i in range(len(cam_extrinsics)) if i in test_view_ids]

    downsample = float(cfg.downsample)

    train_dataset = Colmap_Dataset(  path, train_exr, cam_intrinsics, downsample = downsample, split = train_num ,frames_start_end = frames_start_end)
    test_dataset =  Colmap_Dataset(  path, test_exr,  cam_intrinsics,  downsample =downsample, split = test_view_ids , frames_start_end = frames_start_end )

    if not cfg.skip_blender:    # default: False; using for render camear trajectory from blender
        cam_extrinsics_blender, cam_intrinsics_blender = readCameraFromBlender(cfg.blender_path)

        if cfg.open_single_view:
            total = (min(frames_start_end[1], len(cam_extrinsics_blender)) - frames_start_end[0])
            cam_extrinsics_blender = [cam_extrinsics_blender[int(cfg.single_view_idx)]] * total
        else:
            blender_offset = int(cfg.blender_offset)
            cam_extrinsics_blender = cam_extrinsics_blender[frames_start_end[0] + blender_offset:min(frames_start_end[1] + blender_offset, len(cam_extrinsics_blender))]

        blender_cam_dataset = Colmap_Dataset_from_Blender(path, cam_extrinsics_blender, cam_intrinsics_blender, downsample=downsample,
                                            split=list(range(0, len(cam_extrinsics_blender))),
                                            frames_start_end=frames_start_end)
    else:
        blender_cam_dataset = None

    train_cam_infos = colmap_format_infos(train_dataset, "train")

    val_cam_infos = interpolation_frames(train_cam_infos, frames_start_end, train_dataset)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = cfg.base_path
    pcd = fetchPly(ply_path)

    print("origin points,",pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300,
                           blender_cameras=blender_cam_dataset
                           )

    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_debug=False, undistorted=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])

        Ts = c2ws[:,:3,3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            if not os.path.exists(cam_name):
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            if idx % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()

            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if "small_city_img" in path:
                c2w[-1,-1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if undistorted:
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]], dtype=np.float32)
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

            if is_debug and idx > 50:
                break
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", ply_path=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if ply_path is None:
        ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras

def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        # R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def load_ply( path) :
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)


    colors = np.random.rand(xyz.shape[0], xyz.shape[1])
    normals = np.random.rand(xyz.shape[0], xyz.shape[1])
    return BasicPointCloud(points=xyz, colors=colors, normals=normals)

def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
    )

    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
    )

    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load
    ply_path = os.path.join(datadir, "downsampled_points_50.ply")


    pcd = load_ply(ply_path)




    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info


def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
}