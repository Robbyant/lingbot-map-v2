# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Online streaming with a transactional paged KV cache."""

import numpy as np
import torch
from torch.nn import functional as F

from ..layers.flashinfer_cache import PagedKVCache
from ..layers.rope import apply_rotary_emb
from ..utils.pose_enc import decode_predictions
from .base import finish_global, join_images, pack_tokens, unpack_tokens


def prepare_global(block, tokens, rope):
    batch, frames, patches, channels = tokens.shape
    packed = pack_tokens(tokens, 5)
    frequencies = pack_tokens(rope.reshape(1, 1, frames, patches, -1), 5)
    query, key, value = block.attn.project(block.norm1(packed))
    return (
        packed,
        apply_rotary_emb(query, frequencies),
        apply_rotary_emb(key, frequencies),
        value,
    )


class Stream:
    """One stream owns one model. Only persistent arrivals advance its clock."""

    def __init__(self, model, *, output_device="cpu"):
        self.model = model
        self.output_device = model._output_device(output_device)
        self._pending = []
        self._count = 0
        self._paged = None
        self._camera = None
        self._closed = False

    @property
    def frame_count(self):
        return self._count + len(self._pending)

    @property
    def ready(self):
        return self._count >= self.model.config.scale_frames

    def _check(self):
        if self._closed or self.model._stream is not self:
            raise RuntimeError("stream is closed or no longer owns this model")
        if self.model._native.training:
            raise RuntimeError("model must remain in eval mode")

    @torch.inference_mode()
    def online(self, frame, *, persistent=True):
        self._check()
        if not persistent and not self.ready:
            raise ValueError("the initialization prefix must be persistent")
        if self.frame_count >= self.model.config.max_frames:
            raise ValueError("committed temporal capacity reached; start a new stream")
        if self.model._compiled is not None:
            torch.compiler.cudagraph_mark_step_begin()
        try:
            return self._online(frame, persistent=persistent)
        finally:
            if self._paged is not None:
                self._paged.rollback()

    def _online(self, frame, *, persistent):
        image = self.model.prepare(frame)
        if image.shape[1] != 1:
            raise ValueError("online accepts exactly one frame per call")
        image = image.detach().to(self.model.device, copy=True)
        if not self.ready and len(self._pending) + 1 < self.model.config.scale_frames:
            self._pending.append(image)
            return None
        images = join_images([*self._pending, image])
        count, height, width = images.shape[1], *images.shape[-2:]
        native = self.model._native
        agg = native.aggregator
        needed = {4, 11, 17, 23} if native.depth_head is not None else {23}
        features, prefix_kv = {}, []
        if self._paged is not None:
            self._paged.begin()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
            tokens, spatial, temporal = agg.embed(images, self._count)
            patches, channels = tokens.shape[-2:]
            for layer, block in enumerate(agg.global_blocks):
                frame_tokens = agg.frame_blocks[layer](
                    tokens.reshape(count, patches, channels), pos=spatial
                ).reshape(1, count, patches, channels)
                compiled = self.model._compiled
                if compiled is None:
                    packed, query, key, value = prepare_global(
                        block, frame_tokens, temporal
                    )
                else:
                    packed, query, key, value = compiled[layer][0](
                        frame_tokens, temporal
                    )
                if self._paged is None:
                    output = F.scaled_dot_product_attention(query, key, value)
                    output = output[0].transpose(0, 1)
                    prefix_kv.append(
                        (
                            unpack_tokens(key, count, patches, 5),
                            unpack_tokens(value, count, patches, 5),
                        )
                    )
                else:
                    dtype = self._paged.dtype
                    natural_key = unpack_tokens(key, 1, patches, 5)[0, :, 0].transpose(
                        0, 1
                    )
                    natural_value = unpack_tokens(value, 1, patches, 5)[
                        0, :, 0
                    ].transpose(0, 1)
                    self._paged.write(
                        layer, natural_key.to(dtype), natural_value.to(dtype)
                    )
                    output = self._paged.run(layer, query[0].transpose(0, 1).to(dtype))
                tokens = (
                    finish_global(block, packed, output)
                    if compiled is None
                    else compiled[layer][1](packed, output)
                )
                tokens = unpack_tokens(tokens, count, patches, 5)
                if layer in needed:
                    features[layer] = torch.cat((frame_tokens, tokens), dim=-1)
        with torch.autocast("cuda", enabled=False):
            poses, camera = native.camera_head(features[23].float(), self._camera)
            output = {"pose_enc": poses}
            if native.depth_head is not None:
                depth, confidence = native.depth_head(
                    [features[i].float() for i in (4, 11, 17, 23)],
                    images=images.float(),
                    patch_start_idx=5,
                )
                output.update(depth=depth, depth_conf=confidence)
        result = decode_predictions(
            output, (height, width), output_device=self.output_device
        )
        result["frame_indices"] = (
            np.arange(self._count, self._count + count)
            if self.output_device.type == "cpu"
            else torch.arange(
                self._count, self._count + count, device=self.output_device
            )
        )
        if not persistent:
            return result
        pool = self._paged
        if pool is None:
            key = prefix_kv[0][0]
            pool = PagedKVCache(
                layers=len(prefix_kv),
                heads=key.shape[1],
                dim=key.shape[-1],
                patches=patches - 5,
                scale=self.model.config.scale_frames,
                window=self.model.config.sliding_window,
                capacity=self.model.config.max_frames,
                device=images.device,
            )
            for index in range(count):
                pool.begin()
                for layer, (key, value) in enumerate(prefix_kv):
                    pool.write(
                        layer,
                        key[0, :, index].transpose(0, 1).to(pool.dtype),
                        value[0, :, index].transpose(0, 1).to(pool.dtype),
                    )
                pool.commit()
        else:
            pool.commit()
        self._paged = pool
        self._camera = camera
        self._count += count
        self._pending.clear()
        return result

    def close(self):
        if self._closed:
            return
        self._paged = self._camera = None
        self._pending.clear()
        self._closed = True
        if self.model._stream is self:
            self.model._stream = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_exception):
        self.close()
