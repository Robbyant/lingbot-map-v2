# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License v1
# (LICENSES/FAIR-Noncommercial-Research-License-v1.txt) and the DINOv3 License
# Agreement (LICENSES/DINOv3-License.txt). See LICENSES/THIRD_PARTY.md.
from typing import Callable

from torch import Tensor, nn

from .attention import SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            use_qk_norm=use_qk_norm,
            device=device,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )

        self.sample_drop_ratio = drop_path

    def forward(self, x_or_x_list, rope_or_rope_list=None):
        def block(x, rope):
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            return x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        if isinstance(x_or_x_list, Tensor):
            return block(x_or_x_list, rope_or_rope_list)
        ropes = rope_or_rope_list or [None] * len(x_or_x_list)
        return [block(x, rope) for x, rope in zip(x_or_x_list, ropes)]
