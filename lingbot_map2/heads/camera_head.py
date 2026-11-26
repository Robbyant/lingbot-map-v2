# Modifications copyright 2026 The Robbyant Team Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Camera pose decoding for online inference. See LICENSES/THIRD_PARTY.md.

import torch
from torch import nn

from ..layers.block import Block
from torch.nn import functional as F
from .head_act import activate_pose
from ..layers.mlp import Mlp


class CameraBlock(Block):
    def __init__(self, dim, num_heads):
        super().__init__(dim, num_heads, qk_norm=False)

    def forward(self, x, bank, layer):
        q, k, v = self.attn.project(self.norm1(x))
        # Each iteration has its own chronological Camera cache. Assign fresh
        # tensors, so the enclosing stream can discard a failed transaction.
        for name, value in ((f"k_{layer}", k), (f"v_{layer}", v)):
            value = value.unsqueeze(3)
            old = bank.get(name)
            bank[name] = value if old is None else torch.cat((old, value), dim=2)
        key = bank[f"k_{layer}"].flatten(2, 3)
        value = bank[f"v_{layer}"].flatten(2, 3)
        # Compute FP32 Camera attention with an explicit visibility mask.
        visible = torch.ones(
            x.shape[0], 1, q.shape[2], key.shape[2], dtype=torch.bool, device=x.device
        )
        output = F.scaled_dot_product_attention(q, key, value, attn_mask=visible)
        output = output.transpose(1, 2).reshape(x.shape)
        x = x + self.ls1(self.attn.proj_drop(self.attn.proj(output)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class CameraHead(nn.Module):
    """Four-stage camera pose decoder for incoming frames."""

    def __init__(self, dim_in=2048, num_heads=16, trunk_depth=4):
        super().__init__()
        self.num_iterations = 4
        self.trunk = nn.Sequential(
            *[CameraBlock(dim_in, num_heads) for _ in range(trunk_depth)]
        )
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, 9))
        self.embed_pose = nn.Linear(9, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in, hidden_features=dim_in // 2, out_features=9, drop=0
        )

    def forward(self, features, cache=None):
        tokens = self.token_norm(features[:, :, 0])
        banks = (
            [dict(bank) for bank in cache]
            if cache is not None
            else [{} for _ in range(self.num_iterations)]
        )
        prediction = None
        poses = []
        for iteration, bank in enumerate(banks):
            conditioning = (
                self.empty_pose_tokens.expand(*tokens.shape[:2], -1)
                if prediction is None
                else prediction.detach()
            )
            shift, scale, gate = self.poseLN_modulation(
                self.embed_pose(conditioning)
            ).chunk(3, dim=-1)
            modulated = gate * (self.adaln_norm(tokens) * (1 + scale) + shift)
            modulated = modulated + tokens
            for layer, block in enumerate(self.trunk):
                modulated = block(modulated, bank, layer)
            delta = self.pose_branch(self.trunk_norm(modulated))
            prediction = delta if prediction is None else prediction + delta
            poses.append(
                activate_pose(
                    prediction, trans_act="linear", quat_act="linear", fl_act="relu"
                )
            )
        return poses, banks
