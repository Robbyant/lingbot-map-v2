# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 The Robbyant Team Authors. All rights reserved.
# Adapted from Robbyant/lingbot-map, lingbot_map/vis/glb_export.py
# (b9dd1b1de535e6d36e777fb42b8433c781bfd783). See LICENSES/THIRD_PARTY.md.

"""GLB 3D export utilities for Online and batch reconstruction predictions."""

import os
from typing import Optional, Tuple

import matplotlib
import numpy as np

from .sky_segmentation import apply_sky_segmentation

try:
    import trimesh
except ImportError:
    trimesh = None


def _rot_y(degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _se3_inverse(matrices: np.ndarray) -> np.ndarray:
    """Invert a batch of [N, 4, 4] or [N, 3, 4] rigid transforms."""
    matrices = np.asarray(matrices, dtype=np.float64)
    n = matrices.shape[0]
    rot = matrices[:, :3, :3]
    trans = matrices[:, :3, 3]
    inv = np.broadcast_to(np.eye(4, dtype=np.float64), (n, 4, 4)).copy()
    rot_t = np.swapaxes(rot, -1, -2)
    inv[:, :3, :3] = rot_t
    inv[:, :3, 3] = -np.einsum("nij,nj->ni", rot_t, trans)
    return inv


def _extract_c2w_and_w2c(predictions: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (c2w [S,4,4], w2c [S,4,4]) from either V2 poses_c2w or V1 extrinsic."""
    if predictions.get("poses_c2w") is not None:
        poses = np.asarray(predictions["poses_c2w"], dtype=np.float64)
        c2w = np.broadcast_to(np.eye(4, dtype=np.float64), (len(poses), 4, 4)).copy()
        c2w[:, :3, :4] = poses[:, :3, :4]
        w2c = _se3_inverse(c2w)
        return c2w, w2c
    if predictions.get("extrinsic") is not None:
        ext = np.asarray(predictions["extrinsic"], dtype=np.float64)
        w2c = np.broadcast_to(np.eye(4, dtype=np.float64), (len(ext), 4, 4)).copy()
        w2c[:, :3, :4] = ext[:, :3, :4]
        c2w = _se3_inverse(w2c)
        return c2w, w2c
    raise ValueError("Predictions must contain 'poses_c2w' or 'extrinsic'")


def _unproject_depth_to_world(
    depth: np.ndarray,
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Unproject [S, H, W] or [S, H, W, 1] depth maps to [S, H, W, 3] world points."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth with shape (S, H, W[, 1]), got {depth.shape}")
    num_frames, height, width = depth.shape
    yy, xx = np.mgrid[0:height, 0:width]
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3).astype(np.float64)
    points = np.empty((num_frames, height, width, 3), dtype=np.float32)
    for i in range(num_frames):
        inv_k = np.linalg.inv(np.asarray(intrinsics[i], dtype=np.float64))
        rays = pixels @ inv_k.T
        cam_pts = rays * depth[i].reshape(-1, 1)
        world_pts = cam_pts @ poses_c2w[i, :3, :3].T + poses_c2w[i, :3, 3]
        points[i] = world_pts.reshape(height, width, 3).astype(np.float32)
    return points


def predictions_to_glb(
    predictions: dict,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    target_dir: Optional[str] = None,
    prediction_mode: str = "Predicted Depthmap",
    skyseg_model_path: str = "skyseg.onnx",
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
) -> "trimesh.Scene":
    """Convert predictions to a 3D scene represented as a trimesh.Scene for GLB export."""
    if trimesh is None:
        raise ImportError(
            "trimesh is required for GLB export. Install with: pip install trimesh"
        )
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")
    if conf_thres is None:
        conf_thres = 10.0

    selected_frame_idx = None
    if filter_by_frames not in ("all", "All"):
        try:
            selected_frame_idx = int(str(filter_by_frames).split(":")[0])
        except (ValueError, IndexError):
            pass

    c2w_all, w2c_all = _extract_c2w_and_w2c(predictions)
    intrinsics = predictions.get("intrinsics", predictions.get("intrinsic"))

    if "Pointmap" in prediction_mode and predictions.get("world_points") is not None:
        pred_world_points = np.asarray(predictions["world_points"])
        pred_world_points_conf = np.asarray(
            predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        )
    else:
        if predictions.get("world_points_from_depth") is not None:
            pred_world_points = np.asarray(predictions["world_points_from_depth"])
        else:
            if predictions.get("depth") is None or intrinsics is None:
                raise ValueError(
                    "Depth-based GLB export requires depth, poses_c2w/extrinsic, and intrinsics"
                )
            pred_world_points = _unproject_depth_to_world(
                predictions["depth"], c2w_all, intrinsics
            )
        pred_world_points_conf = np.asarray(
            predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
        )

    images = np.asarray(predictions["images"])

    if mask_sky:
        pred_world_points_conf = _apply_sky_mask(
            pred_world_points_conf,
            target_dir,
            images,
            skyseg_model_path=skyseg_model_path,
            sky_mask_dir=sky_mask_dir,
            sky_mask_visualization_dir=sky_mask_visualization_dir,
        )

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        c2w_all = c2w_all[selected_frame_idx][None]
        w2c_all = w2c_all[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)

    if images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    colors_rgb = colors_rgb.reshape(-1, 3)
    if colors_rgb.dtype != np.uint8:
        scale = 255.0 if colors_rgb.size > 0 and colors_rgb.max() <= 1.0 else 1.0
        colors_rgb = np.clip(colors_rgb * scale, 0, 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    valid = np.isfinite(vertices_3d).all(axis=1) & np.isfinite(conf)
    conf_threshold = np.percentile(conf[valid], conf_thres) if conf_thres > 0 and valid.any() else 0.0
    conf_mask = valid & (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        conf_mask &= colors_rgb.sum(axis=1) >= 16
    if mask_white_bg:
        conf_mask &= ~(
            (colors_rgb[:, 0] > 240)
            & (colors_rgb[:, 1] > 240)
            & (colors_rgb[:, 2] > 240)
        )

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d.size == 0:
        vertices_3d = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        colors_rgb = np.array([[255, 255, 255]], dtype=np.uint8)
        scene_scale = 1.0
    else:
        lower = np.percentile(vertices_3d, 5, axis=0)
        upper = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = max(float(np.linalg.norm(upper - lower)), 0.1)

    colormap = matplotlib.colormaps["gist_rainbow"]

    scene_3d = trimesh.Scene()
    scene_3d.add_geometry(trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb))

    num_cameras = len(c2w_all)
    if show_cam:
        for i in range(num_cameras):
            rgba_color = colormap(i / max(num_cameras, 1))
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            integrate_camera_into_scene(scene_3d, c2w_all[i], current_color, scene_scale)

    return apply_scene_alignment(scene_3d, w2c_all)


def _apply_sky_mask(
    conf: np.ndarray,
    target_dir: Optional[str],
    images: np.ndarray,
    skyseg_model_path: str = "skyseg.onnx",
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
) -> np.ndarray:
    image_folder = None
    if target_dir is not None:
        candidate = os.path.join(target_dir, "images")
        if os.path.isdir(candidate):
            image_folder = candidate
        if sky_mask_dir is None:
            sky_mask_dir = os.path.join(target_dir, "sky_masks")

    return apply_sky_segmentation(
        conf,
        image_folder=image_folder,
        images=images,
        skyseg_model_path=skyseg_model_path,
        sky_mask_dir=sky_mask_dir,
        sky_mask_visualization_dir=sky_mask_visualization_dir,
    )


def integrate_camera_into_scene(
    scene: "trimesh.Scene",
    transform: np.ndarray,
    face_colors: Tuple[int, int, int],
    scene_scale: float,
    frustum_thickness: float = 1.0,
):
    """Integrate a camera frustum mesh into the 3D scene."""
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = _rot_z(45.0)
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = _rot_z(2.0)

    shell_scales = [1.0, 0.95]
    shell_transforms = [np.eye(4), slight_rotation]
    if frustum_thickness > 1.0:
        n_extra = max(1, int(frustum_thickness - 1))
        for k in range(1, n_extra + 1):
            angle = 2.0 + k * 2.0
            scale = 1.0 + k * 0.02
            rot = np.eye(4)
            rot[:3, :3] = _rot_z(angle)
            shell_scales.append(scale)
            shell_transforms.append(rot)
            rot_neg = np.eye(4)
            rot_neg[:3, :3] = _rot_z(-angle)
            shell_scales.append(scale)
            shell_transforms.append(rot_neg)

    vertices_parts = [
        transform_points(t_mat, s * camera_cone_shape.vertices)
        for s, t_mat in zip(shell_scales, shell_transforms)
    ]
    vertices_combined = np.concatenate(vertices_parts)
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces_multi(camera_cone_shape, len(shell_scales))
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def build_trajectory_tube(positions, radius, colormap, num_cameras):
    """Build a gradient-colored cylinder tube mesh connecting camera positions."""
    if trimesh is None:
        return None
    segments = []
    for i in range(len(positions) - 1):
        p0, p1 = positions[i], positions[i + 1]
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < 1e-8:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=seg_len, sections=8)
        direction = (p1 - p0) / seg_len
        mid = (p0 + p1) / 2.0

        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(z_axis, direction)
        c = float(np.dot(z_axis, direction))
        if np.linalg.norm(v) < 1e-8:
            rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array(
                [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
            )
            rot = np.eye(3) + vx + vx @ vx / (1.0 + c)

        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = mid
        cyl.apply_transform(transform)

        t_color = (i + 0.5) / max(num_cameras - 1, 1)
        rgba = colormap(t_color)
        cyl.visual.face_colors[:, :3] = tuple(int(255 * x) for x in rgba[:3])
        segments.append(cyl)

    if not segments:
        return None
    return trimesh.util.concatenate(segments)


def apply_scene_alignment(
    scene_3d: "trimesh.Scene",
    extrinsics_matrices: np.ndarray,
) -> "trimesh.Scene":
    """Bring world coordinates into camera 0's frame (W2C), then apply display axes."""
    opengl_conversion_matrix = get_opengl_conversion_matrix()
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = _rot_y(180.0)

    initial_transformation = (
        align_rotation @ opengl_conversion_matrix @ extrinsics_matrices[0]
    )
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """Return the OpenGL conversion matrix (flips Y and Z axes)."""
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(
    transformation: np.ndarray,
    points: np.ndarray,
    dim: Optional[int] = None,
) -> np.ndarray:
    """Apply a 4x4 transformation matrix to a set of 3D points."""
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces_multi(
    cone_shape: "trimesh.Trimesh", num_shells: int
) -> np.ndarray:
    """Compute faces for a camera mesh with multiple shells (for thicker frustums)."""
    faces_list = []
    nv = len(cone_shape.vertices)
    for s in range(num_shells - 1):
        off_a = s * nv
        off_b = (s + 1) * nv
        for face in cone_shape.faces:
            if 0 in face:
                continue
            v1, v2, v3 = face
            faces_list.extend(
                [
                    (v1 + off_a, v2 + off_a, v2 + off_b),
                    (v1 + off_a, v1 + off_b, v3 + off_a),
                    (v3 + off_b, v2 + off_a, v3 + off_a),
                ]
            )
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)
