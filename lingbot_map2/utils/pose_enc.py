# Modifications copyright 2026 The Robbyant Team Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. See LICENSE.txt and LICENSES/THIRD_PARTY.md.

import numpy as np
import torch
from .rotation import quat_to_mat


def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,
    pose_encoding_type="absT_quaR_FoV",
    build_intrinsics=True,  # e.g., (256, 512)
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
            where B is batch size and S is sequence length.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera pose matrices with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing the
              camera-to-world (C2W) transformation `[R|t]` in this model (mapping
              camera-frame points to world coordinates, matching `poses_c2w` returned
              by `decode_predictions`).
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(
                pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device
            )
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    elif pose_encoding_type == "absT_quaR":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        intrinsics = None

    return extrinsics, intrinsics


def decode_predictions(output, image_hw, *, output_device="cpu"):
    """Decode native pose and geometry outputs for a completed streaming step."""
    height, width = image_hw
    encoding = output["pose_enc"]
    if isinstance(encoding, (list, tuple)):
        encoding = encoding[-1]
    encoding = encoding.float()
    frames = encoding.shape[1]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(encoding, (height, width))
    # Native pose encoding is C2W. No inverse, GT alignment or rescaling.
    poses = torch.eye(4, device=extrinsics.device).repeat(frames, 1, 1)
    poses[:, :3] = extrinsics[0]
    result = {
        "poses_c2w": poses,
        "intrinsics": intrinsics[0],
        "pose_enc": encoding[0],
    }
    for key in ("depth", "depth_conf"):
        if key in output:
            result[key] = output[key][0].float()
    target = poses.device if output_device is None else torch.device(output_device)
    if target.type == "cpu":
        result = {key: value.cpu().numpy() for key, value in result.items()}
        finite = all(np.isfinite(value).all() for value in result.values())
    else:
        result = {key: value.to(target) for key, value in result.items()}
        # One scalar validation barrier, without transferring full predictions.
        finite = bool(
            torch.stack([torch.isfinite(x).all() for x in result.values()]).all()
        )
    if not finite:
        raise RuntimeError("model produced non-finite predictions")
    return result
