# Modifications copyright 2026 The Robbyant Team Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. See LICENSE.txt and LICENSES/THIRD_PARTY.md.

"""Image loading, preprocessing and deterministic keyframe selection."""

from __future__ import annotations

from collections.abc import Sequence

import json
import math
import re
from pathlib import Path


def uniform_keyframe_ids(count, *, prefix=8, interval=None):
    """Keep the prefix, then every ceil(N/320)-th frame unless overridden.

    This is a nominal budget, not an exact 320-frame cap. IDs address the input
    sequence; the model's committed-frame clock is a separate coordinate.
    """
    if type(count) is not int or type(prefix) is not int or not 0 < prefix <= count:
        raise ValueError("input count must accommodate a positive prefix")
    if interval is None:
        interval = math.ceil(count / 320)
    if type(interval) is not int or interval < 1:
        raise ValueError("keyframe interval must be a positive integer")
    return list(range(prefix)) + list(range(prefix, count, interval))


def exact_keyframe_ids(count, *, prefix=8, num_keyframes=320):
    """Keep all frames up to num_keyframes, or keep the prefix plus uniformly
    spaced frames so the persistent KV-cache budget equals num_keyframes.

    When count > num_keyframes:
    - num_keyframes == prefix retains only the initialization prefix [0..prefix-1].
    - num_keyframes == prefix + 1 retains the prefix plus the final frame count - 1.
    - num_keyframes >= prefix + 2 retains the prefix plus uniformly spaced frames
      spanning [prefix, count - 1] (inclusive).
    """
    if type(count) is not int or type(prefix) is not int or not 0 < prefix <= count:
        raise ValueError("input count must accommodate a positive prefix")
    if type(num_keyframes) is not int or num_keyframes < prefix:
        raise ValueError(f"num_keyframes must be an integer >= prefix ({prefix})")
    if count <= num_keyframes:
        return list(range(count))
    remaining = num_keyframes - prefix
    if remaining == 0:
        return list(range(prefix))
    if remaining == 1:
        return list(range(prefix)) + [count - 1]
    span = count - 1 - prefix
    denom = remaining - 1
    # Exact half-up integer rounding guarantees strictly increasing indices
    # spanning [prefix, count - 1] since count > num_keyframes implies span > denom.
    tail = [prefix + (2 * i * span + denom) // (2 * denom) for i in range(remaining)]
    return list(range(prefix)) + tail


def image_paths(folder: str | None, image_list: str | None) -> list[Path]:
    if image_list:
        source = Path(image_list).resolve(strict=True)
        values = json.loads(source.read_text())
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) for value in values)
        ):
            raise ValueError("image-list must be a non-empty JSON array of paths")
        paths = [(source.parent / value).resolve(strict=True) for value in values]
    else:
        root = Path(folder).resolve(strict=True)

        def natural_key(path):
            return [
                (1, int(part)) if part.isdigit() else (0, part.lower())
                for part in re.split(r"(\d+)", path.name)
            ]

        paths = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ),
            key=natural_key,
        )
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("input must contain readable image files")
    return paths


def cover_resize(rgb, output_wh):
    """Keep input dtype, cover the target, then center crop without padding."""
    import cv2

    width, height = output_wh
    raw_height, raw_width = rgb.shape[:2]
    scale = max(width / raw_width, height / raw_height)
    resized_width, resized_height = round(raw_width * scale), round(raw_height * scale)
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    left, top = (resized_width - width) // 2, (resized_height - height) // 2
    return resized[top : top + height, left : left + width].copy()


def load_images(paths: Sequence[str | Path], width: int = 512, height: int = 288):
    """RGB float32 AREA cover-resize / center-crop, then divide by 255.

    Preserve channels-last tensor strides and floating-point resized pixels.
    """
    import numpy as np
    import torch
    from PIL import Image

    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("width and height must be positive multiples of 16")
    if not paths:
        raise ValueError("no input images")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        frames.append(cover_resize(rgb, (width, height)))
    return (torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255)[None]
