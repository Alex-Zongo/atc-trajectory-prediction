"""Shared training helpers."""

import numpy as np
import torch


def pick_device(device=None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_by_leg(meta, seed: int = 0, val_frac: float = 0.1):
    """Hold out whole LEGS, never individual windows.

    A leg is atomic: splitting inside one would put consecutive points of the same
    trajectory on both sides, and the model would be scored on state it had effectively
    seen. Returns a boolean mask that is True for validation rows.
    """
    legs = np.asarray(meta["leg_id"].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    val = set(rng.choice(legs, size=max(1, int(val_frac * len(legs))), replace=False))
    return meta["leg_id"].isin(val).values
