# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Public API for the pure causal LingBot-Map2 streaming release."""

from dataclasses import replace
from functools import partial
from pathlib import Path
import numpy as np
import torch

from ..utils.checkpoint import resolve_checkpoint
from ..utils.compilation import _cached_encoder_rope, _cached_depth_position
from ..utils.load_fn import cover_resize, load_images


class GCRStream:
    """Online reconstruction model with one active stream."""

    def __init__(self, native, config, metadata=None, *, width=None, height=None):
        self._native = native
        self.config = config
        self.metadata = metadata or {}
        self.width = config.width if width is None else width
        self.height = config.height if height is None else height
        if any(
            type(x) is not int or x <= 0 or x % 16 for x in (self.width, self.height)
        ):
            raise ValueError("width and height must be positive multiples of 16")
        self._stream = None
        self._compiled = None

    @property
    def device(self):
        return next(self._native.parameters()).device

    def prepare(self, frames):
        """Paths, one uint8 HWC RGB array, or already preprocessed float32 CHW tensors.

        Tensors must already have the configured geometry. No implicit resize,
        range conversion, EXIF rotation or layout conversion is applied to them.
        """
        if isinstance(frames, (str, Path)):
            frames = [frames]
        if isinstance(frames, np.ndarray):
            if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[-1] != 3:
                raise ValueError("array input must be one uint8 HWC RGB frame")
            rgb = cover_resize(frames.astype(np.float32), (self.width, self.height))
            frames = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255)[None, None]
        elif not torch.is_tensor(frames):
            frames = load_images(list(frames), self.width, self.height)
        if frames.ndim == 3:
            frames = frames[None, None]
        elif frames.ndim == 4:
            frames = frames[None]
        if (
            frames.ndim != 5
            or frames.shape[0] != 1
            or frames.shape[2] != 3
            or tuple(frames.shape[-2:]) != (self.height, self.width)
            or frames.shape[1] == 0
            or frames.dtype != torch.float32
        ):
            raise ValueError(
                "expected float32 [1, N, 3, height, width] at configured geometry"
            )
        valid = (
            torch.isfinite(frames).all()
            & (frames.min() >= -1e-6)
            & (frames.max() <= 1 + 1e-6)
        )
        if not bool(valid):
            raise ValueError("expected finite RGB in [0, 1]")
        return frames

    def _output_device(self, output_device):
        target = self.device if output_device is None else torch.device(output_device)
        if (
            target.type == "cuda"
            and target.index is None
            and self.device.type == "cuda"
        ):
            target = self.device
        if target.type != "cpu" and target != self.device:
            raise ValueError("output_device must be cpu or the model device")
        return target

    def compile(self):
        """Compile pure streaming compute; page transactions remain explicit."""
        if self._stream is not None:
            raise RuntimeError("close the active stream before enabling compilation")
        if self.device.type != "cuda":
            raise ValueError("compilation requires a CUDA model")
        if self._compiled is not None:
            return self
        from ..aggregator.stream import prepare_global
        from ..aggregator.base import finish_global

        agg = self._native.aggregator
        for block in [*agg.frame_blocks, *agg.patch_embed.blocks]:
            block.forward = torch.compile(block.forward, mode="reduce-overhead")
        if agg.patch_embed.rope_embed is not None:
            rope = agg.patch_embed.rope_embed
            rope.forward = _cached_encoder_rope(rope)
        if self._native.depth_head is not None:
            head = self._native.depth_head
            head._apply_pos_embed = _cached_depth_position(head)
        self._compiled = [
            (
                torch.compile(partial(prepare_global, block), mode="reduce-overhead"),
                torch.compile(partial(finish_global, block), mode="reduce-overhead"),
            )
            for block in agg.global_blocks
        ]
        self.metadata["compile"] = "reduce-overhead; lazy shape-dependent capture"
        return self

    def create_stream(self, *, output_device="cpu"):
        from ..aggregator.stream import Stream

        if self._stream is not None:
            raise RuntimeError("close the active stream before creating another")
        stream = Stream(self, output_device=output_device)
        self._stream = stream
        return stream

    def warmup(self, frames, *, output_device="cpu"):
        """Use disposable streams on real geometry before measuring throughput."""
        if self._stream is not None:
            raise RuntimeError("close the active stream before warmup")
        images = self.prepare(frames).to(self.device)
        if not self.config.scale_frames < images.shape[1] <= self.config.max_frames:
            raise ValueError(
                "warmup needs the full prefix plus at least one online frame"
            )
        passes = 3 if self._compiled is not None else 1
        for _ in range(passes):
            with self.create_stream(output_device=output_device) as stream:
                for frame in images[0]:
                    stream.online(frame)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.metadata["last_warmup"] = {
            "frames": images.shape[1],
            "passes": passes,
            "input_wh": [self.width, self.height],
        }


def load_model(
    checkpoint,
    *,
    config=None,
    device="cuda:0",
    depth=True,
    width=None,
    height=None,
    compile=True,
    max_frames=None,
):
    """Load the published causal weights/config pair. No default weights exist."""
    path, spec, config_path = resolve_checkpoint(checkpoint, config)
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("streaming inference requires a BF16-capable CUDA GPU")
    with torch.cuda.device(target):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this model requires BF16 support")
    for value in (width, height):
        if value is not None and (type(value) is not int or value <= 0 or value % 16):
            raise ValueError("width and height must be positive multiples of 16")
    from .gcr_base import NativeModel

    capacity = spec.max_frames if max_frames is None else max_frames
    if type(capacity) is not int or capacity < spec.scale_frames:
        raise ValueError("max_frames must accommodate the initialization prefix")
    native = NativeModel(spec.spatial_rope, max_frames=capacity)
    # Only model-only tensor exports are supported; never unpickle training objects.
    checkpoint_data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint_data, dict):
        raise ValueError("expected a model state dictionary or a model-only export")
    if checkpoint_data.get("training_mode", "causal") != "causal":
        raise ValueError("checkpoint metadata contradicts the causal configuration")
    state = checkpoint_data.get("model", checkpoint_data)
    native.load_state_dict(state, strict=True)
    del state, checkpoint_data
    # Audit every head before applying an explicitly requested pose-only option.
    if not depth:
        native.depth_head = None
    native.eval().to(target)
    metadata = {
        "release": "LingBot-Map2 Streaming (causal-only)",
        "checkpoint_sha256": spec.sha256,
        "checkpoint_config": spec.to_dict(),
        "checkpoint_config_path": str(config_path),
        "temporal_capacity": capacity,
        "depth_enabled": depth,
        "attention": "SDPA initialization/frame attention; FlashInfer FA2 paged online",
        "precision": "BF16 backbone; FP32 Camera4 and DPT",
        "compile": False,
    }
    model = GCRStream(
        native, replace(spec, max_frames=capacity), metadata, width=width, height=height
    )
    return model.compile() if compile else model
