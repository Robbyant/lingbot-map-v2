# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""DINOv3 image embedding and geometric context token layout."""

import torch
from torch import nn

from ..layers.block import Block
from ..layers.dinov3 import build_encoder
from ..layers.rope import (
    DinoV3RotaryPositionEmbedding2D,
    PositionGetter,
    WanRotaryPosEmbed,
)


class Aggregator(nn.Module):
    def __init__(self, spatial_rope="max", max_frames=1024):
        super().__init__()
        self.patch_embed = build_encoder(patch_size=16, embed_dim=1024)
        self.patch_embed.mask_token.requires_grad_(False)
        self.rope = DinoV3RotaryPositionEmbedding2D(base=100)
        self.position_getter = PositionGetter()
        self.frame_blocks = nn.ModuleList(
            [Block(1024, 16, rope=self.rope) for _ in range(24)]
        )
        self.global_blocks = nn.ModuleList(
            [Block(1024, 16, rope=self.rope) for _ in range(24)]
        )
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, 1024))
        self.register_token = nn.Parameter(torch.empty(1, 2, 4, 1024))
        self.rope3d = WanRotaryPosEmbed(
            attention_head_dim=64,
            patch_size=(max_frames, 32, 32),
            max_seq_len=max(1024, max_frames),
            theta=10000.0,
            spatial_normalize_coords=spatial_rope,
        )
        for name, value in (
            ("_resnet_mean", [0.485, 0.456, 0.406]),
            ("_resnet_std", [0.229, 0.224, 0.225]),
        ):
            self.register_buffer(
                name, torch.tensor(value).view(1, 1, 3, 1, 1), persistent=False
            )

    def embed(self, images, start):
        _, frames, _, height, width = images.shape
        pixels = ((images - self._resnet_mean) / self._resnet_std).view(
            frames, 3, height, width
        )
        patches = self.patch_embed(pixels)
        if isinstance(patches, dict):
            patches = patches["x_norm_patchtokens"]
        special = []
        for template in (self.camera_token, self.register_token):
            if start == 0:
                values = torch.cat(
                    (template[:, :1], template[:, 1:].expand(1, frames - 1, -1, -1)),
                    dim=1,
                )
            else:
                values = template[:, 1:].expand(1, frames, -1, -1)
            special.append(values.reshape(frames, -1, patches.shape[-1]))
        tokens = torch.cat((*special, patches), dim=1)
        positions = (
            self.position_getter(
                frames, height // 16, width // 16, device=images.device
            )
            + 1
        )
        positions = torch.cat(
            (
                torch.zeros(frames, 5, 2, dtype=positions.dtype, device=images.device),
                positions,
            ),
            dim=1,
        )
        rope = self.rope3d(
            frames,
            height // 16,
            width // 16,
            5,
            device=images.device,
            f_start=start,
            f_end=start + frames,
        )
        return tokens, positions, rope


def join_images(images):
    """Join arrivals without turning native HWC-backed inputs into CHW storage."""
    if len(images) == 1:
        return images[0]
    if all(image.stride(-3) == 1 for image in images):
        return torch.cat(
            [image.permute(0, 1, 3, 4, 2) for image in images], dim=1
        ).permute(0, 1, 4, 2, 3)
    return torch.cat(images, dim=1)


def pack_tokens(tokens, special):
    return torch.cat(
        (
            tokens[..., special:, :].flatten(-3, -2),
            tokens[..., :special, :].flatten(-3, -2),
        ),
        dim=-2,
    )


def unpack_tokens(tokens, frames, patches, special):
    boundary = frames * (patches - special)
    return torch.cat(
        (
            tokens[..., boundary:, :].unflatten(-2, (frames, special)),
            tokens[..., :boundary, :].unflatten(-2, (frames, patches - special)),
        ),
        dim=-2,
    )


def finish_global(block, x, output):
    """Pure projection and FFN residuals, shared by eager and compiled online."""
    attention = block.attn
    output = output[None].transpose(1, 2)
    output = output.transpose(1, 2).reshape(x.shape)
    x = x + block.ls1(attention.proj_drop(attention.proj(output)))
    x = x + block.ls2(block.mlp(block.norm2(x)))
    return x
