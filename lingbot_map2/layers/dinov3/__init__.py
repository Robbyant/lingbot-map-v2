# Modifications copyright 2026 The Robbyant Team Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License v1
# (LICENSES/FAIR-Noncommercial-Research-License-v1.txt) and the DINOv3 License
# Agreement (LICENSES/DINOv3-License.txt). See LICENSES/THIRD_PARTY.md.

from .attention import LinearKMaskedBias, SelfAttention
from .block import SelfAttentionBlock
from .ffn_layers import Mlp, SwiGLUFFN
from .layer_scale import LayerScale
from .patch_embed import PatchEmbed
from .rms_norm import RMSNorm
from .rope_position_encoding import RopePositionEmbedding

__all__ = [
    "LayerScale",
    "LinearKMaskedBias",
    "Mlp",
    "PatchEmbed",
    "RMSNorm",
    "RopePositionEmbedding",
    "SelfAttention",
    "SelfAttentionBlock",
    "SwiGLUFFN",
]


from .vision_transformer import DinoVisionTransformer


def build_encoder(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    """Build the DINOv3 ViT-L/16 image encoder."""
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model
