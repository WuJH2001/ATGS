import torch
import tinycudann as tcnn
import torch.nn as nn
import os
from arguments.atgs_cfg import cfg
from scene.spacetime_utils import routing_encoder_id as compute_routing_encoder_id
from scene.spacetime_utils import num_routed_encoders as compute_num_routed_encoders


class SpaceTimeHashingField(torch.nn.Module):
    def __init__(self, xyz_bound_min, xyz_bound_max, hashmap_size=16, activation="ReLU", n_levels=16,
                 n_features_per_level=4, base_resolution=16, n_neurons=128, feat_dim=64, levels=13):
        super(SpaceTimeHashingField, self).__init__()

        self.feat_dim = feat_dim
        self.enc_models = nn.ModuleList()
        self.levels = int(cfg.levels)
        levels = self.levels

        if self.levels == 0:
            enc_model = tcnn.NetworkWithInputEncoding(
                n_input_dims=4,
                n_output_dims=64,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 4,
                    "log2_hashmap_size": 20,
                    "base_resolution": 16,
                    "per_level_scale": 2.0,
                },
                # 100M
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": activation,
                    "output_activation": "ReLU",
                    "n_neurons": n_neurons,
                    "n_hidden_layers": 1,
                },
            )
            self.enc_models.append(enc_model)
        else:
            if cfg.open_clear_storage_hash:
                for level in range(levels):
                    if True:
                        enc_model = tcnn.NetworkWithInputEncoding(
                            n_input_dims=4,
                            # n_output_dims=64 - int(cfg.level_0_dim),  # 16    64  as same as 4dgs
                            n_output_dims=self.feat_dim // 2 if cfg.open_feat_cat else self.feat_dim,
                            encoding_config={
                                "otype": "HashGrid",
                                "n_levels": 16,  # 16
                                "n_features_per_level": 8,  # 8
                                "log2_hashmap_size": 19,  # 19
                                "base_resolution": 16,  #
                                "per_level_scale": 2.0,
                            },

                            network_config={
                                "otype": "FullyFusedMLP",
                                "activation": activation,
                                "output_activation": "ReLU",
                                "n_neurons": n_neurons,
                                "n_hidden_layers": 1,
                            },
                        )
                        
                    self.enc_models.append(enc_model)

                    # for idx in range(2**level):
                    #     enc_model = tcnn.NetworkWithInputEncoding(
                    #         n_input_dims = 4,
                    #         n_output_dims = 64,    # 16    64  as same as 4dgs
                    #         encoding_config={
                    #             "otype": "HashGrid" ,
                    #             "n_levels": 16 , # 16
                    #             "n_features_per_level": 8,  # 8
                    #             "log2_hashmap_size": 19 , # 20
                    #             "base_resolution": 16, #
                    #             "per_level_scale": 2.0 ,
                    #         },

                    #         network_config={
                    #             "otype": "FullyFusedMLP",
                    #             "activation": activation,
                    #             "output_activation": "ReLU",
                    #             "n_neurons": n_neurons,
                    #             "n_hidden_layers": 1 ,
                    #         },
                    #     )
                    #     self.enc_models.append(enc_model)
            else:
                for level in range(levels + 1):
                    if level == 0:
                        enc_model = tcnn.NetworkWithInputEncoding(
                            n_input_dims=4,
                            n_output_dims=int(cfg.level_0_dim),
                            encoding_config={
                                "otype": "HashGrid",
                                "n_levels": 16,
                                "n_features_per_level": 4,
                                "log2_hashmap_size": 20,
                                "base_resolution": 16,
                                "per_level_scale": 2.0,
                            },
                            # 100M
                            network_config={
                                "otype": "FullyFusedMLP",
                                "activation": activation,
                                "output_activation": "ReLU",
                                "n_neurons": n_neurons,
                                "n_hidden_layers": 1,
                            },
                        )
                    else:
                        enc_model = tcnn.NetworkWithInputEncoding(
                            n_input_dims=4,
                            # n_output_dims=64 - int(cfg.level_0_dim),  # 16    64  as same as 4dgs
                            n_output_dims=self.feat_dim // 2 if cfg.open_feat_cat else self.feat_dim,
                            encoding_config={
                                "otype": "HashGrid",
                                "n_levels": 16,  # 16
                                "n_features_per_level": 8,  # 8
                                "log2_hashmap_size": 19,  # 19
                                "base_resolution": 16,  #
                                "per_level_scale": 2.0,
                            },

                            network_config={
                                "otype": "FullyFusedMLP",
                                "activation": activation,
                                "output_activation": "ReLU",
                                "n_neurons": n_neurons,
                                "n_hidden_layers": 1,
                            },
                        )
                        
                    self.enc_models.append(enc_model)


        self.mlp_mask = tcnn.Network(
            n_input_dims=4,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 1,
            },
        )

        self.register_buffer('xyz_bound_min', xyz_bound_min)
        self.register_buffer('xyz_bound_max', xyz_bound_max)
        self.clear_storage = bool(cfg.open_clear_storage_hash)

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

    def forward(self, xyz: torch.Tensor, time, old_time):
        contracted_xyz = self.get_contracted_xyz(xyz)  # Shape: [N, 3]

        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)

        dynamic_features = []
        time_scalar = float(time[0].item())

        old_time_scalar = float(old_time[0].item())
        # for i in range(self.levels):
        #     idx = (2**i - 1) + int(time_scalar * 2**i)

        #     time_i = (time[mask] * (2**i) - int(time_scalar * 2**i) )
        #     hash_inputs = torch.cat([contracted_xyz[mask], time_i], dim=-1) # time_i

        #     dynamic_feature_level = self.enc_models[idx](hash_inputs)  # [M, feat_dim]
        # dynamic_feature_out = torch.cat(dynamic_features, dim=-1)
        # dynamic_feature_out = sum(dynamic_features)

        batch_size = int(cfg.hash_batch_size)
        num_masked = contracted_xyz[mask].shape[0]


        if self.levels == 0:
            hash_inputs_0 = torch.cat([contracted_xyz[mask], time[mask]], dim=-1)
            dynamic_feature_0_list = []
            
            for i in range(0, num_masked, batch_size):
                end_idx = min(i + batch_size, num_masked)
                batch_inputs = hash_inputs_0[i:end_idx]
                batch_output = self.enc_models[0](batch_inputs)
                dynamic_feature_0_list.append(batch_output)
            
            dynamic_feature_0 = torch.cat(dynamic_feature_0_list, dim=0)

        if self.levels != 0:
            time_i = time[mask] * self.levels - int(time_scalar * self.levels)
            hash_inputs_1 = torch.cat([contracted_xyz[mask], time_i], dim=-1)
            dynamic_feature_1_list = []

            for i in range(0, num_masked, batch_size):
                end_idx = min(i + batch_size, num_masked)
                batch_inputs = hash_inputs_1[i:end_idx]
                if cfg.open_clear_storage_hash:
                    batch_output = self.enc_models[int(old_time_scalar * self.levels)](batch_inputs)
                else:
                    batch_output = self.enc_models[int(old_time_scalar * self.levels) + 1](batch_inputs)
                dynamic_feature_1_list.append(batch_output)

            dynamic_feature_1 = torch.cat(dynamic_feature_1_list, dim=0)
            # dynamic_feature_out = torch.cat([dynamic_feature_0, dynamic_feature_1], dim=-1)
            dynamic_feature_out = dynamic_feature_1
            hash_inputs = hash_inputs_1
        else:
            dynamic_feature_out = dynamic_feature_0
            hash_inputs = hash_inputs_0

        if cfg.open_feat_cat:
            dynamic_feature = torch.zeros((xyz.shape[0], self.feat_dim // 2), device="cuda")
        else:
            dynamic_feature = torch.zeros((xyz.shape[0], self.feat_dim), device="cuda")
        dynamic_feature[mask] = dynamic_feature_out.float()

        batch_size = int(cfg.hash_batch_size)
        num_masked = hash_inputs.shape[0]
        temp_dynamics_list = []

        for i in range(0, num_masked, batch_size):
            end_idx = min(i + batch_size, num_masked)
            batch_inputs = hash_inputs[i:end_idx]
            batch_output = self.mlp_mask(batch_inputs)
            temp_dynamics_list.append(batch_output)

        temp_dynamics = torch.cat(temp_dynamics_list, dim=0)
        dynmiac_mask = torch.zeros((xyz.shape[0], 1), device="cuda")
        dynmiac_mask[mask] = torch.sigmoid(temp_dynamics.float())

        return dynamic_feature, dynmiac_mask

    def get_params(self):
        open_split_dy_lr = cfg.open_split_dy_lr

        if open_split_dy_lr:
            param_dict = dict(self.named_parameters())
            return_list = []

            for name in param_dict.keys():
                t = {}
                t['params'] = param_dict[name]
                t['name'] = name

                return_list.append(t)

            return return_list
        else:
            parameter_list = []
            for name, param in self.named_parameters():
                parameter_list.append(param)
            return parameter_list

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():  # enc_model.para
            if "enc_model" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_hash_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "enc_model" in name:
                parameter_list.append(param)

        return parameter_list
