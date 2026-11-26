# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License v1
# (LICENSES/FAIR-Noncommercial-Research-License-v1.txt) and the DINOv3 License
# Agreement (LICENSES/DINOv3-License.txt). See LICENSES/THIRD_PARTY.md.
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer(
                "bias_mask", torch.full_like(self.bias, fill_value=math.nan)
            )

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = (
            self.bias * self.bias_mask.to(self.bias.dtype)
            if self.bias is not None
            else None
        )
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(
        self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, rope=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        output = F.scaled_dot_product_attention(q, k, v)
        return self.proj_drop(self.proj(output.transpose(1, 2).reshape(B, N, C)))
