# Modifications copyright 2026 The Robbyant Team Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Transformer blocks for online inference. See LICENSES/THIRD_PARTY.md.

import torch
from torch import nn
from torch.nn import functional as F

from .layer_scale import LayerScale
from .mlp import Mlp


class Attention(nn.Module):
    def __init__(self, dim, num_heads, *, qk_norm=False, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.rope = rope

    def project(self, x):
        batch, tokens, _ = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        return self.q_norm(q), self.k_norm(k), v

    def forward(self, x, pos=None):
        q, k, v = self.project(x)
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        output = F.scaled_dot_product_attention(q, k, v)
        return self.proj_drop(self.proj(output.transpose(1, 2).reshape(x.shape)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, *, qk_norm=True, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qk_norm=qk_norm, rope=rope)
        self.ls1 = LayerScale(dim, init_values=0.01)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4)
        self.ls2 = LayerScale(dim, init_values=0.01)

    def forward(self, x, pos=None):
        x = x + self.ls1(self.attn(self.norm1(x), pos))
        return x + self.ls2(self.mlp(self.norm2(x)))
