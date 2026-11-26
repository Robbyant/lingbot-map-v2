# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Checkpoint-native modules shared by the streaming model."""

from torch import nn

from ..aggregator.base import Aggregator
from ..heads.camera_head import CameraHead
from ..heads.dpt_head import DPTHead


class NativeModel(nn.Module):
    def __init__(self, spatial_rope="max", max_frames=1024):
        super().__init__()
        self.aggregator = Aggregator(spatial_rope, max_frames=max_frames)
        self.camera_head = CameraHead()
        self.depth_head = DPTHead(
            dim_in=2048,
            patch_size=16,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        )
