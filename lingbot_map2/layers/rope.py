# Modifications copyright 2026 The Robbyant Team Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. See LICENSE.txt and LICENSES/THIRD_PARTY.md.

import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple, Optional, Union


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return (
            cached_positions.view(1, height * width, 2)
            .expand(batch_size, -1, -1)
            .clone()
        )


class DinoV3RotaryPositionEmbedding2D(nn.Module):
    """Checkpoint-native DINOv3 RoPE for frame-local aggregator attention.

    Patch centres use max(H, W) normalization. PositionGetter positions are
    offset by one; special tokens at (0, 0) are left unrotated.
    The positional embedding has no learned parameters.
    """

    def __init__(self, base: float = 100.0):
        super().__init__()
        self.base = base

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        dim = tokens.size(-1)
        if dim % 4:
            raise ValueError("DINOv3 RoPE requires head dimension divisible by four")
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape (batch, tokens, 2)")
        is_patch = (positions > 0).all(dim=-1)
        max_hw = positions.amax(dim=(1, 2)).clamp(min=1).float()
        coords = 2.0 * (positions.float() - 0.5) / max_hw[:, None, None] - 1.0
        periods = self.base ** (
            2
            * torch.arange(dim // 4, device=tokens.device, dtype=torch.float32)
            / (dim // 2)
        )
        angles = (2 * np.pi * coords[..., None] / periods).flatten(-2).tile(2)
        cos = torch.where(is_patch[..., None], angles.cos(), 1.0)[:, None]
        sin = torch.where(is_patch[..., None], angles.sin(), 0.0)[:, None]
        x = tokens.float()
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos + rotated * sin).to(tokens.dtype)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """Construct real or complex rotary phases at the supplied positions."""

    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # [S]

    theta = theta * ntk_factor

    freqs = (
        1.0
        / (
            theta
            ** (
                torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[
                    : (dim // 2)
                ]
                / dim
            )
        )
        / linear_factor
    )

    freqs = torch.outer(pos, freqs)  # [S, D/2]

    if use_real and repeat_interleave_real:
        freqs_cos = (
            freqs.cos()
            .repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
            .float()
        )  # [S, D]
        freqs_sin = (
            freqs.sin()
            .repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
            .float()
        )  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64: [S, D/2]
        return freqs_cis


class WanRotaryPosEmbed(nn.Module):
    """Temporal and spatial RoPE with checkpoint-native phase construction."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        fhw_dim: Optional[Tuple[int, int, int]] = [20, 22, 22],
        complex_dtype: torch.dtype = torch.complex128,
        spatial_normalize_coords: str = "none",
    ):
        super().__init__()

        if spatial_normalize_coords not in ("none", "max"):
            raise ValueError("spatial_normalize_coords must be 'none' or 'max'")
        self.spatial_normalize_coords = spatial_normalize_coords
        self.theta = theta

        if complex_dtype not in (torch.complex64, torch.complex128):
            raise ValueError(
                "complex_dtype must be torch.complex64 or torch.complex128, "
                f"got {complex_dtype}"
            )

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        if fhw_dim is not None:
            assert attention_head_dim == sum(fhw_dim), (
                f"attention_head_dim {attention_head_dim} must match sum(fhw_dim) {sum(fhw_dim)}"
            )
            t_dim, h_dim, w_dim = fhw_dim
        else:
            h_dim = w_dim = 2 * (attention_head_dim // 6)
            t_dim = attention_head_dim - h_dim - w_dim

        self.fhw_dim = (t_dim, h_dim, w_dim)

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=False,
                repeat_interleave_real=False,
                freqs_dtype=torch.float64,
            )
            freqs.append(freq)

        # Generate phases in float64 above, then optionally store them as complex64.
        # The latter halves runtime RoPE storage and enables float32 complex math while
        # preserving the higher-precision phase construction.
        self.freqs = torch.cat(freqs, dim=1).to(complex_dtype)

    def forward(
        self,
        ppf,
        pph,
        ppw,
        patch_start_idx,
        device: torch.device,
        f_start: int = 0,
        f_end: Optional[int] = None,
    ) -> torch.Tensor:
        """Build frame-major phases for five special tokens and a spatial patch grid."""

        self.freqs = self.freqs.to(device)

        if hasattr(self, "fhw_dim") and self.fhw_dim is not None:
            t_dim, h_dim, w_dim = self.fhw_dim
        else:
            h_dim = w_dim = 2 * (self.attention_head_dim // 6)
            t_dim = self.attention_head_dim - h_dim - w_dim

        freqs = self.freqs.split_with_sizes(
            [
                t_dim // 2,
                h_dim // 2,
                w_dim // 2,
            ],
            dim=1,
        )

        if f_end is not None:
            ppf = f_end - f_start
            frame_slice = slice(f_start, f_end)
        else:
            frame_slice = slice(0, ppf)

        max_pos = max(
            f_end if f_end is not None else ppf, max(pph, ppw) + patch_start_idx
        )
        if max_pos > self.max_seq_len:
            raise RuntimeError(
                f"WanRotaryPosEmbed: requested position {max_pos} exceeds "
                f"max_seq_len={self.max_seq_len}. Increase max_frames when loading the model "
                f"(ppf={ppf}, pph={pph}, ppw={ppw}, patch_start_idx={patch_start_idx})."
            )

        if self.spatial_normalize_coords == "max":
            # Normalize patch centres only, without the special-token offset.
            # Time and special-token phases use their indexed frequencies.
            spatial_freqs = []
            for size, dim in ((pph, h_dim), (ppw, w_dim)):
                coords = torch.arange(size, dtype=torch.float64, device=device)
                coords = 2.0 * (coords + 0.5) / max(pph, ppw) - 1.0
                spatial_freqs.append(
                    get_1d_rotary_pos_embed(
                        dim,
                        coords,
                        self.theta,
                        use_real=False,
                        freqs_dtype=torch.float64,
                    ).to(self.freqs.dtype)
                )
            patch_freqs_h, patch_freqs_w = spatial_freqs
        else:
            # Keep the default path bitwise identical, including its offset.
            patch_freqs_h = freqs[1][patch_start_idx : patch_start_idx + pph]
            patch_freqs_w = freqs[2][patch_start_idx : patch_start_idx + ppw]

        ## For other tokens
        if patch_start_idx > 0:
            # camera: (f, 0, 0); registers 0..3: (f, 1, 1) through (f, 4, 4)
            # Shape: (ppf, patch_start_idx, dim)
            freqs_special_f = (
                freqs[0][frame_slice]
                .reshape(ppf, 1, -1)
                .expand(ppf, patch_start_idx, -1)
            )
            freqs_special_h = (
                freqs[1][:patch_start_idx]
                .reshape(1, patch_start_idx, -1)
                .expand(ppf, patch_start_idx, -1)
            )
            freqs_special_w = (
                freqs[2][:patch_start_idx]
                .reshape(1, patch_start_idx, -1)
                .expand(ppf, patch_start_idx, -1)
            )
            freqs_special = torch.cat(
                [freqs_special_f, freqs_special_h, freqs_special_w], dim=-1
            )
            freqs_special = freqs_special.reshape(
                ppf, patch_start_idx, -1
            )  # (ppf, patch_start_idx, dim)

            # Indexed spatial patches use the special-token offset; max mode instead
            # uses the normalized patch centres computed above.
            # Shape: (ppf, pph, ppw, dim)
            freqs_f = (
                freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
            )
            freqs_h = patch_freqs_h.reshape(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
            freqs_w = patch_freqs_w.reshape(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
            freqs_patches = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)
            freqs_patches = freqs_patches.reshape(ppf, pph * ppw, -1)

            # Concatenate special tokens and patches for each frame along the second dimension
            # Shape: (ppf, patch_start_idx + pph * ppw, dim)
            freqs = torch.cat(
                [freqs_special, freqs_patches], dim=1
            )  # (ppf, patch_start_idx + pph * ppw, dim)

            # Flatten to get final shape: (ppf * (patch_start_idx + pph * ppw), dim)
            freqs = freqs.reshape(ppf * (patch_start_idx + pph * ppw), -1)
            freqs = freqs.unsqueeze(0).unsqueeze(0)
            return freqs

        freqs_f = freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = patch_freqs_h.reshape(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = patch_freqs_w.reshape(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(
            1, 1, ppf * pph * ppw, -1
        )  # (1, 1, ppf * pph * ppw, dim)
        return freqs


def apply_rotary_emb(x, freqs):
    """Apply complex rotary phases to the paired channels of Q or K."""

    if freqs.dtype == torch.complex128:
        work_dtype = torch.float64
    elif freqs.dtype == torch.complex64:
        work_dtype = torch.float32
    else:
        raise TypeError(f"freqs must be complex64 or complex128, got {freqs.dtype}")
    x_reshaped = x.to(work_dtype).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)

    x_complex = torch.view_as_complex(x_reshaped)

    x_rotated = x_complex * freqs.to(device=x.device, dtype=x_complex.dtype)

    x_real = torch.view_as_real(x_rotated)

    x_out = x_real.flatten(3)

    return x_out.to(x.dtype)
