"""
Runtime ATGS config shared across modules that do not receive argparse args.

Populate via apply_atgs_cfg(args) after CLI + config-file merge (see train.py / render.py).
Defaults match arguments/vrugz/basketball.py; override there for experiments.
"""


class ATGSCfg:
    # ----- point cloud / time -----
    base_path = ""
    time_window_size = 5
    frame_interval = 1
    downsample = 1.0
    llffhold = 10

    # ----- representation -----
    hash = True
    levels = 13
    level_0_dim = 32
    use_hashgrid = False
    open_feat_cat = True
    open_clear_storage_hash = True
    open_split_dy_lr = False
    open_anchor_feat_delay = True
    enable_lifetime_decay = False
    lifetime_min_decay = 0.5
    mlp_chunk_size = 0
    max_lr_iteration = 200_000
    hash_batch_size = 1024 * 256
    auto_encoder_levels = False
    encoder_balanced_accumulation = False

    # ----- render / training helpers (formerly env vars) -----
    primitive_type = "3dgs"
    near_plane = 0.01
    absgrad = False
    dy_lr_scale = 1.0
    clip_size = 20
    render_feature = "all"
    render_kind = "smooth"
    virtual_frame_interval = 40
    save_optimizer = True

    skip_blender = True
    blender_path = ""
    open_single_view = False
    single_view_idx = 730
    blender_offset = 0
    blender_name = "main"

    open_sync_time = False

    # ----- derived from --frames_start_end / -m -----
    model_path = ""
    total_frames = 0
    frame_start_idx = 0
    frame_end_idx = 0
    total_point_frames = 0


cfg = ATGSCfg()

_APPLY_KEYS = [
    "base_path",
    "time_window_size",
    "frame_interval",
    "downsample",
    "llffhold",
    "hash",
    "levels",
    "level_0_dim",
    "use_hashgrid",
    "open_feat_cat",
    "open_clear_storage_hash",
    "open_split_dy_lr",
    "open_anchor_feat_delay",
    "enable_lifetime_decay",
    "lifetime_min_decay",
    "mlp_chunk_size",
    "max_lr_iteration",
    "hash_batch_size",
    "auto_encoder_levels",
    "encoder_balanced_accumulation",
    "primitive_type",
    "near_plane",
    "absgrad",
    "dy_lr_scale",
    "clip_size",
    "render_feature",
    "render_kind",
    "virtual_frame_interval",
    "save_optimizer",
    "skip_blender",
    "blender_path",
    "open_single_view",
    "single_view_idx",
    "blender_offset",
    "blender_name",
    "open_sync_time",
]


def apply_atgs_cfg(args):
    """Copy merged argparse/config values onto the shared cfg singleton."""
    for key in _APPLY_KEYS:
        if hasattr(args, key):
            setattr(cfg, key, getattr(args, key))

    cfg.model_path = getattr(args, "model_path", "")
    start, end = args.frames_start_end
    cfg.frame_start_idx = int(start)
    cfg.frame_end_idx = int(end)
    cfg.total_frames = int(end) - int(start)
    interval = max(int(cfg.frame_interval), 1)
    cfg.total_point_frames = cfg.total_frames // interval

    if cfg.auto_encoder_levels:
        clip_size = max(int(cfg.clip_size), 1)
        cfg.levels = max((cfg.total_frames + clip_size - 1) // clip_size, 1)
        args.levels = cfg.levels
