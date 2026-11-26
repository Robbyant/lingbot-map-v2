# Modifications copyright 2026 The Robbyant Team Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. See LICENSE.txt and LICENSES/THIRD_PARTY.md.

import torch


def _cached_encoder_rope(module):
    """Cache spatial frequencies for inference."""
    original = module.forward
    cached_key = cached_value = None

    def forward(*, H, W):
        nonlocal cached_key, cached_value
        periods = module.periods
        if module.training or torch.is_inference(periods):
            return original(H=H, W=W)
        key = (
            H,
            W,
            periods.device,
            periods.dtype,
            periods.data_ptr(),
            periods._version,
            module.dtype,
            module.normalize_coords,
        )
        if key != cached_key:
            cached_value = original(H=H, W=W)
            cached_key = key
        return cached_value

    return forward


def _cached_depth_position(module):
    """Reuse native eval-only DPT grids for the current geometry and device."""
    from ..heads.utils import create_uv_grid, position_grid_to_embed

    original = module._apply_pos_embed
    geometry = None
    embeddings = {}

    def forward(x, W, H, ratio=0.1):
        nonlocal geometry
        if module.training or torch.is_grad_enabled():
            return original(x, W, H, ratio)
        current = (W, H, x.device, x.dtype)
        if current != geometry:
            embeddings.clear()
            geometry = current
        channels, height, width = x.shape[1:]
        key = (channels, height, width, ratio)
        if key not in embeddings:
            grid = create_uv_grid(
                width, height, aspect_ratio=W / H, dtype=x.dtype, device=x.device
            )
            embedding = position_grid_to_embed(grid, channels) * ratio
            embeddings[key] = embedding.permute(2, 0, 1)[None]
        return x + embeddings[key]

    return forward
