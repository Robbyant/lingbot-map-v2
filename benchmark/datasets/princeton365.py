# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Read prepared or official Princeton365 validation scanning sequences."""

from pathlib import Path

import numpy as np


def read_princeton365(root, scene):
    """Convert reference-camera C2W to user-camera C2W at the original frame IDs."""
    import cv2

    from .adapters import tum_by_frame

    root = Path(root)
    prepared = root / scene / f"{scene}.gt_user_c2w.npy"
    if prepared.exists():
        return root / scene / f"{scene}.mp4", np.load(prepared, allow_pickle=False)

    scanning = root
    if (root / "Benchmark_blur_val/scanning_videos").is_dir():
        scanning = root / "Benchmark_blur_val/scanning_videos"
    if (scanning / "user_view").is_dir():
        candidates = sorted((scanning / "user_view").glob(f"*/{scene}.mp4"))
        if len(candidates) != 1:
            raise ValueError(
                f"{scene}: expected one official user-view video, found {len(candidates)}"
            )
        video = candidates[0]
        trajectory = (
            scanning
            / "gt_view/camera_poses"
            / (scene.replace("_user_view_", "_") + ".txt")
        )
        transform = video.parent / "relative_gt_to_user_optim.npy"
    else:
        # The official extracted WebDataset release stores sidecars per sequence.
        names = [scene, scene.replace("_user_view_", "_")]
        candidates = [
            parent / name / f"{name}.mp4"
            for parent in (root, root / "validation")
            for name in names
            if (parent / name / f"{name}.mp4").is_file()
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{scene}: expected a prepared scene, Benchmark_blur_val/scanning_videos, "
                "or an official extracted validation sequence"
            )
        video = candidates[0]
        trajectory = video.with_suffix(".gt_trajectory.txt")
        transform = video.with_suffix(".relative_transform.npy")

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open RGB video: {video}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count < 8:
        raise ValueError(f"{scene}: video must contain at least eight frames")
    user_to_gt = np.load(transform, allow_pickle=False)
    if (
        user_to_gt.shape != (4, 4)
        or not np.isfinite(user_to_gt).all()
        or not np.allclose(user_to_gt[3], [0, 0, 0, 1])
        or not np.allclose(
            user_to_gt[:3, :3].T @ user_to_gt[:3, :3], np.eye(3), atol=1e-5
        )
        or not np.isclose(np.linalg.det(user_to_gt[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError(f"Invalid user-to-reference camera transform: {transform}")
    # Some official videos end one frame before their GT log. Keep every RGB
    # frame and omit that final GT-only sample; never interpolate or pad RGB.
    poses = tum_by_frame(trajectory, count + 1)[:count]
    valid = np.isfinite(poses).all((1, 2))
    # Same right multiplication as the official Princeton365 evaluator.
    poses[valid] = poses[valid] @ user_to_gt
    return video, poses
