import torch
import torch.nn as nn
from scene.spacetime_hexplane import HexPlaneField
from scene.spacetime_utils import routing_encoder_id as compute_routing_encoder_id
from scene.spacetime_utils import num_routed_encoders as compute_num_routed_encoders
from arguments.atgs_cfg import cfg


class SpaceTimePlaneField(torch.nn.Module):
    def __init__(self, args, xyz_bound_max, xyz_bound_min, feat_dim):
        super(SpaceTimePlaneField, self).__init__()

        self.feat_dim = feat_dim
        self.enc_models = nn.ModuleList()
        self.levels = int(cfg.levels)
        self.clear_storage = bool(cfg.open_clear_storage_hash)
        levels = self.levels

        if levels == 0:
            enc_model = HexPlaneField(args.bounds, args.kplanes_config, args.multires, 64).to("cuda")
            enc_model.set_aabb(xyz_bound_max, xyz_bound_min)
            self.enc_models.append(enc_model)
        elif self.clear_storage:
            out_dim = self.feat_dim // 2 if cfg.open_feat_cat else self.feat_dim
            for _ in range(levels):
                enc_model = HexPlaneField(args.bounds, args.kplanes_config, args.multires, out_dim).to("cuda")
                enc_model.set_aabb(xyz_bound_max, xyz_bound_min)
                self.enc_models.append(enc_model)
        else:
            for level in range(levels + 1):
                if level == 0:
                    enc_model = HexPlaneField(
                        args.bounds, args.kplanes_config, args.multires, int(cfg.level_0_dim)
                    ).to("cuda")
                    enc_model.set_aabb(xyz_bound_max, xyz_bound_min)
                else:
                    enc_model = HexPlaneField(
                        args.bounds, args.kplanes_config, args.multires, 64 - int(cfg.level_0_dim)
                    ).to("cuda")
                    enc_model.set_aabb(xyz_bound_max, xyz_bound_min)
                self.enc_models.append(enc_model)

        self.register_buffer('xyz_bound_min', xyz_bound_min)
        self.register_buffer('xyz_bound_max', xyz_bound_max)

    def routing_encoder_id(self, time_scalar):
        return compute_routing_encoder_id(time_scalar, self.levels, self.clear_storage)

    def num_routed_encoders(self):
        return compute_num_routed_encoders(self.levels, self.clear_storage)

    def dump(self, path):
        torch.save(self.state_dict(), path)

    def get_contracted_xyz(self, xyz):
        with torch.no_grad():
            contracted_xyz = (xyz - self.xyz_bound_min) / (self.xyz_bound_max - self.xyz_bound_min)
            return contracted_xyz

    def forward(self, xyz: torch.Tensor, time, old_time=None):
        if old_time is None:
            old_time = time

        time_scalar = float(time[0].item())
        old_time_scalar = float(old_time[0].item())

        if self.levels == 0:
            dynamic_feature_out, _ = self.enc_models[0](xyz, time)
        elif self.clear_storage:
            encoder_id = self.routing_encoder_id(old_time_scalar)
            time_i = time * self.levels - int(time_scalar * self.levels)
            dynamic_feature_out, _ = self.enc_models[encoder_id](xyz, time_i)
        else:
            dynamic_feature_0, _ = self.enc_models[0](xyz, time)
            time_i = time * self.levels - int(time_scalar * self.levels)
            dynamic_feature_1, _ = self.enc_models[int(time_scalar * self.levels) + 1](xyz, time_i)
            dynamic_feature_out = torch.cat([dynamic_feature_0, dynamic_feature_1], dim=-1)

        dynmiac_mask = torch.zeros((xyz.shape[0], 1), device=xyz.device)
        return dynamic_feature_out, dynmiac_mask

    def get_params(self):
        parameter_list = []
        for name, param in self.named_parameters():
            parameter_list.append(param)
        return parameter_list

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" in name:
                parameter_list.append(param)
        return parameter_list
