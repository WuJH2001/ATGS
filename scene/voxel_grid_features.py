import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn

import os
class VoxelGridFeatures(nn.Module):
    """
    体素网格特征模块：
      - mode = "voxel"（use_hashgrid=False）:
          静态特征用 dense 3D voxel grid + 三线性插值
      - mode = "hash"（use_hashgrid=True）:
          静态特征用 3D 多分辨率 HashGrid + MLP
    """

    def __init__(
        self,
        xyz_bound_min,
        xyz_bound_max,
        grid_resolution=128,
        static_feat_dim=64,
        use_hashgrid: bool = True,
        hash_n_levels=16,
        hash_n_features_per_level=4,
        hash_log2_hashmap_size=19,
        hash_base_resolution=16,
        hash_per_level_scale=2.0,
        hash_mlp_hidden_neurons=64,
        hash_mlp_hidden_layers=1,
    ):
        super(VoxelGridFeatures, self).__init__()

        self.grid_resolution = grid_resolution
        self.static_feat_dim = static_feat_dim
        self.use_hashgrid = use_hashgrid

        self.register_buffer('xyz_bound_min', xyz_bound_min)
        self.register_buffer('xyz_bound_max', xyz_bound_max)

        if not self.use_hashgrid:
            self.mode = "voxel"
            self.static_features = nn.Parameter(
                torch.zeros(grid_resolution, grid_resolution, grid_resolution, static_feat_dim),
                requires_grad=True
            )

        else:
            self.mode = "hash"

            encoding_config = {
                "otype": "HashGrid",
                "n_levels": hash_n_levels,
                "n_features_per_level": hash_n_features_per_level,
                "log2_hashmap_size": hash_log2_hashmap_size,
                "base_resolution": hash_base_resolution,
                "per_level_scale": hash_per_level_scale,
            }

            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hash_mlp_hidden_neurons,
                "n_hidden_layers": hash_mlp_hidden_layers,
            }

            # xyz ∈ [0,1]^3 -> static_feat_dim
            self.hash_encoder = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=static_feat_dim,
                encoding_config=encoding_config,
                network_config=network_config,
            )

    def _normalize_xyz_01(self, xyz):
        """
        将 xyz 归一化到 [0, 1]，供 HashGrid 使用
        """
        return (xyz - self.xyz_bound_min) / (self.xyz_bound_max - self.xyz_bound_min)

    def _normalize_xyz_voxel(self, xyz):
        """
        将 xyz 坐标归一化到 [0, grid_resolution-1] 范围，供 dense voxel 插值使用
        """
        normalized = self._normalize_xyz_01(xyz)              # [0, 1]
        grid_coords = normalized * (self.grid_resolution - 1) # [0, res-1]
        return grid_coords

    def trilinear_interpolate(self, features_grid, grid_coords):
        """
        三线性插值
        Args:
            features_grid: [res, res, res, feat_dim] 特征网格
            grid_coords: [N, 3] 网格坐标 (范围 [0, res-1])
        Returns:
            interpolated_features: [N, feat_dim] 插值后的特征
        """
        N = grid_coords.shape[0]
        res = self.grid_resolution

        grid_coords = torch.clamp(grid_coords, 0, res - 1 - 1e-5)

        grid_coords_floor = torch.floor(grid_coords).long()      # [N, 3]
        grid_coords_frac = grid_coords - grid_coords_floor.float()  # [N, 3]

        x0, y0, z0 = grid_coords_floor[:, 0], grid_coords_floor[:, 1], grid_coords_floor[:, 2]
        x1 = torch.clamp(x0 + 1, 0, res - 1)
        y1 = torch.clamp(y0 + 1, 0, res - 1)
        z1 = torch.clamp(z0 + 1, 0, res - 1)

        xd, yd, zd = grid_coords_frac[:, 0:1], grid_coords_frac[:, 1:2], grid_coords_frac[:, 2:3]

        c000 = features_grid[x0, y0, z0]  # [N, feat_dim]
        c001 = features_grid[x0, y0, z1]
        c010 = features_grid[x0, y1, z0]
        c011 = features_grid[x0, y1, z1]
        c100 = features_grid[x1, y0, z0]
        c101 = features_grid[x1, y0, z1]
        c110 = features_grid[x1, y1, z0]
        c111 = features_grid[x1, y1, z1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - zd) + c1 * zd

        return c

    def query_features(self, xyz):
        """
        查询给定位置的特征

        Args:
            xyz: [N, 3] 世界坐标

        Returns:
            static_feat: [N, static_feat_dim]
        """
        if self.mode == "voxel":
            grid_coords = self._normalize_xyz_voxel(xyz)
            static_feat = self.trilinear_interpolate(self.static_features, grid_coords)
        else:
            # 3D HashGrid
            normalized_xyz = self._normalize_xyz_01(xyz)         # [0, 1]^3
            static_feat = self.hash_encoder(normalized_xyz).float()

        return static_feat

    def get_static_parameters(self):
        """
        返回当前模式下需要优化的静态参数：
          - voxel 模式：dense 体素参数
          - hash 模式：HashGrid + MLP 参数
        """
        if self.mode == "voxel":
            return [self.static_features]
        else:
            return list(self.hash_encoder.parameters())

