# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 The Robbyant Team Authors. All rights reserved.
# Adapted from Robbyant/lingbot-map, lingbot_map/vis/utils.py
# (b9dd1b1de535e6d36e777fb42b8433c781bfd783). See LICENSES/THIRD_PARTY.md.

"""Visualization utility functions for camera states, colorization and color bars."""

import dataclasses
from typing import Optional, Tuple

import cv2
import matplotlib
import numpy as np


def _get_colormap(cmap_name: str):
    return matplotlib.colormaps[cmap_name]


@dataclasses.dataclass
class CameraState:
    """Camera state for rendering."""

    fov: float
    aspect: float
    c2w: np.ndarray

    def get_K(self, img_wh: Tuple[int, int]) -> np.ndarray:
        """Get camera intrinsic matrix from FOV and image size."""
        width, height = img_wh
        focal_length = height / 2.0 / np.tan(self.fov / 2.0)
        return np.array(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )


def get_vertical_colorbar(
    h: int,
    vmin: float,
    vmax: float,
    cmap_name: str = "jet",
    label: Optional[str] = None,
    cbar_precision: int = 2,
) -> np.ndarray:
    """Create a vertical colorbar image as an (H, W, 3) float32 array in [0, 1]."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)
    cmap = _get_colormap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_loc = np.linspace(vmin, vmax, 6)
    cb1 = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical"
    )

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)
    cb1.ax.tick_params(labelsize=18, rotation=0)
    if label is not None:
        cb1.set_label(label)

    canvas.draw()
    buf, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(buf, np.uint8).reshape((height, width, 4))
    im = im[:, :, :3].astype(np.float32) / 255.0

    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(
    x: np.ndarray,
    cmap_name: str = "jet",
    mask: Optional[np.ndarray] = None,
    range: Optional[Tuple[float, float]] = None,
    append_cbar: bool = False,
    cbar_in_image: bool = False,
    cbar_precision: int = 2,
) -> np.ndarray:
    """Turn a grayscale image [H, W] into a color image [H, W, 3]."""
    x = np.array(x, copy=True)
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        x[np.logical_not(mask)] = vmin
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += 1e-6

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / max(vmax - vmin, 1e-12)

    cmap = _get_colormap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask_f = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask_f + np.ones_like(x_new) * (1.0 - mask_f)

    if append_cbar:
        cbar = get_vertical_colorbar(
            h=x.shape[0],
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            cbar_precision=cbar_precision,
        )
        if cbar_in_image:
            x_new[:, -cbar.shape[1] :, :] = cbar
        else:
            x_new = np.concatenate(
                (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1
            )
    return x_new


def colorize(
    x,
    cmap_name: str = "jet",
    mask=None,
    range: Optional[Tuple[float, float]] = None,
    append_cbar: bool = False,
    cbar_in_image: bool = False,
):
    """Turn a grayscale PyTorch tensor [H, W] or [B, H, W] into a color tensor."""
    import torch

    device = x.device
    x_np = x.detach().cpu().numpy()
    if mask is not None:
        mask_np = mask.detach().cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)
    else:
        mask_np = None
        kernel = None

    if x_np.ndim == 2:
        x_np = x_np[None]
        if mask_np is not None:
            mask_np = mask_np[None]

    out = []
    for idx, frame in enumerate(x_np):
        frame_mask = None
        if mask_np is not None:
            frame_mask = cv2.erode(
                mask_np[idx].astype(np.uint8), kernel, iterations=1
            ).astype(bool)
        colored = colorize_np(
            frame, cmap_name, frame_mask, range, append_cbar, cbar_in_image
        )
        out.append(torch.from_numpy(colored).to(device).float())
    return torch.stack(out).squeeze(0)
