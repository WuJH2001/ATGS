from arguments.atgs_cfg import cfg

encoder_visit_history = {}


def init_encoder_visit_history():
    levels = int(cfg.levels)
    for level in range(levels):
        encoder_visit_history[str(level)] = 0
