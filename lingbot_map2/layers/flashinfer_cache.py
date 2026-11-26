# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Paged patch/special KV storage for the native Map2 streaming runtime."""

import collections
import math

import torch


class PagedKVCache:
    """An all-layer append transaction with prefix, window and special-token pages.

    One physical page per frame's patch tokens; all chronological special tokens
    occupy a packed tail. Only page IDs change on eviction or rollback. One free
    patch page and uncommitted special slots protect live state from failed heads.
    Metadata and one FlashInfer plan are shared across all layers.
    """

    def __init__(
        self,
        *,
        layers,
        heads,
        dim,
        patches,
        special=5,
        scale=8,
        window=64,
        capacity=1024,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        import flashinfer

        if min(layers, heads, dim, patches, special, scale, window) <= 0:
            raise ValueError(
                "paged cache dimensions, prefix and window must be positive"
            )
        if capacity < scale:
            raise ValueError("capacity must accommodate the initialization prefix")
        self.layers, self.heads, self.dim = layers, heads, dim
        self.patches, self.special, self.scale, self.window = (
            patches,
            special,
            scale,
            window,
        )
        self.capacity = capacity
        self.device = torch.device(device)
        self.dtype = dtype
        # One extra patch page makes failed/transient steps independent of eviction.
        self.patch_pages = scale + window + 1
        self.special_pages = math.ceil((capacity + 1) * special / patches)
        self.pool = [
            torch.empty(
                (self.patch_pages + self.special_pages, 2, patches, heads, dim),
                device=device,
                dtype=dtype,
            )
            for _ in range(layers)
        ]
        self.active = collections.deque()
        self.free = list(range(self.patch_pages - 1, -1, -1))
        self.count = 0
        self.pending = None
        self.workspace = torch.empty(64 << 20, dtype=torch.uint8, device=device)
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend="fa2"
        )

    def begin(self):
        if self.pending is not None:
            raise RuntimeError("nested append")
        if self.count >= self.capacity:
            raise ValueError("keyframe capacity reached")
        page = self.free[-1]
        active = list(self.active) + [(self.count, page)]
        active = [
            (i, p)
            for i, p in active
            if i < self.scale or i >= max(self.scale, self.count + 1 - self.window)
        ]
        tokens = (self.count + 1) * self.special
        pages = [p for _, p in active] + list(
            range(self.patch_pages, self.patch_pages + math.ceil(tokens / self.patches))
        )
        self.pending = {"page": page, "active": active, "written": set()}
        self.wrapper.plan(
            torch.tensor(
                [0, self.patches + self.special], dtype=torch.int32, device=self.device
            ),
            torch.tensor([0, len(pages)], dtype=torch.int32, device=self.device),
            torch.tensor(pages, dtype=torch.int32, device=self.device),
            torch.tensor(
                [tokens % self.patches or self.patches],
                dtype=torch.int32,
                device=self.device,
            ),
            self.heads,
            self.heads,
            self.dim,
            self.patches,
            causal=False,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def write(self, layer, k, v):
        """[patches+special, heads, dim], specials first, post-RoPE K."""
        if self.pending is None or layer in self.pending["written"]:
            raise RuntimeError("invalid write")
        if k.dtype != self.dtype or v.dtype != self.dtype:
            raise TypeError("no silent KV conversion")
        for bank, x in enumerate((k, v)):
            self.pool[layer][self.pending["page"], bank].copy_(x[self.special :])
            start = self.count * self.special
            done = 0
            while done < self.special:
                p, off = divmod(start + done, self.patches)
                take = min(self.special - done, self.patches - off)
                self.pool[layer][self.patch_pages + p, bank, off : off + take].copy_(
                    x[done : done + take]
                )
                done += take
        self.pending["written"].add(layer)

    def run(self, layer, q):
        if layer not in self.pending["written"]:
            raise RuntimeError("query before write")
        if q.dtype != self.dtype:
            raise TypeError("query dtype mismatch")
        return self.wrapper.run(q.contiguous(), self.pool[layer])

    def commit(self):
        if self.pending is None or len(self.pending["written"]) != self.layers:
            raise RuntimeError("incomplete append")
        page = self.free.pop()
        assert page == self.pending["page"]
        self.active.append((self.count, page))
        self.count += 1
        while len(self.active) > self.scale + self.window:
            index, page = self.active[self.scale]
            del self.active[self.scale]
            self.free.append(page)
        self.pending = None

    def rollback(self):
        # Writes affected only a free patch page and invisible special tail slots.
        self.pending = None
