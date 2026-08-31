"""Shared routing helpers for interchangeable spacetime encoders (hash / plane)."""


def routing_encoder_id(time_scalar, levels, clear_storage=True):
    """Map normalized time in [0, 1) to the active encoder index."""
    levels = int(levels)
    if levels <= 0:
        return 0
    chunk_id = min(max(int(float(time_scalar) * levels), 0), levels - 1)
    if clear_storage:
        return chunk_id
    return chunk_id + 1


def num_routed_encoders(levels, clear_storage=True):
    """Number of time-routed encoders used by balanced sampling / accumulation."""
    levels = int(levels)
    if levels <= 0:
        return 1
    return levels if clear_storage else levels + 1
