# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Shared-ROI reconstruction scoring: one Sim(3), voxelization, rigid ICP."""

import numpy as np

from .pose import sim3


def validate_reference(path, raw_ids, protocol):
    with np.load(path, allow_pickle=False) as data:
        required = {
            "raw_ids",
            "input_wh",
            "mask",
            "alignment_mask",
            "points",
            "c2w",
            "source_id",
        }
        if protocol["dataset"] != "oxford":
            required.add("pixel_xy")
        if required - set(data.files):
            raise ValueError(f"reference is missing {required - set(data.files)}")
        np.testing.assert_array_equal(data["raw_ids"], raw_ids)
        np.testing.assert_array_equal(data["input_wh"], protocol["input_wh"])
        w, h = protocol["geometry"]["grid_wh"]
        mask = data["mask"]
        if mask.dtype != bool or mask.shape != (len(raw_ids), h, w) or mask.sum() < 3:
            raise ValueError("invalid shared GT mask")
        alignment = data["alignment_mask"]
        if (
            alignment.dtype != bool
            or alignment.shape != mask.shape
            or (alignment & ~mask).any()
        ):
            raise ValueError(
                "alignment mask must be a subset of the common evaluation mask"
            )
        points = data["points"]
        if protocol["dataset"] == "euroc":
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or not np.isfinite(points).all()
            ):
                raise ValueError(
                    "EuRoC reference must be the finite visible scan union"
                )
        elif points.shape != (*mask.shape, 3) or not np.isfinite(points[mask]).all():
            raise ValueError(
                "reference must provide finite pixel-corresponding GT points"
            )
        poses = data["c2w"]
        if poses.shape != (len(raw_ids), 4, 4) or not np.isfinite(poses).all():
            raise ValueError("dense reference requires valid associated C2W poses")
        if protocol["dataset"] != "oxford":
            xy = data["pixel_xy"]
            if xy.shape != (h, w, 2) or not np.issubdtype(xy.dtype, np.integer):
                raise ValueError(
                    "pixel_xy must be integer scoring-grid to input lookup"
                )
            used = xy[mask.any(0)]
            if (used < 0).any() or (used >= protocol["input_wh"]).any():
                raise ValueError("reference ROI exceeds prediction footprint")
        if not str(data["source_id"].item()):
            raise ValueError("reference provenance is required")


def predicted_points(depth, intrinsics, poses, reference, protocol):
    """Project all shared GT pixels; never use prediction confidence/range masks."""
    w, h = protocol["geometry"]["grid_wh"]
    parts = []
    for i, pose in enumerate(poses):
        mask = reference["mask"][i]
        native_depth = np.asarray(depth[i]).squeeze()
        if protocol["dataset"] == "oxford":
            import cv2

            nh, nw = native_depth.shape
            z = cv2.resize(native_depth, (w, h), interpolation=cv2.INTER_NEAREST)[mask]
            yy, xx = np.mgrid[:h, :w]
            xy = np.stack([xx[mask], yy[mask]], -1)
            k = intrinsics[i].copy()
            k[0] *= w / nw
            k[1] *= h / nh
        else:
            xy = reference["pixel_xy"][mask]
            z = native_depth[xy[:, 1], xy[:, 0]]
            k = intrinsics[i]
        if not np.isfinite(z).all() or (z <= 0).any():
            raise ValueError("invalid prediction depth in the fixed shared ROI")
        pixels = np.column_stack([xy, np.ones(len(xy))])
        points = ((pixels @ np.linalg.inv(k).T) * z[:, None]) @ pose[:3, :3].T + pose[
            :3, 3
        ]
        parts.append(points.astype(np.float32))
    return np.concatenate(parts)


def score_clouds(
    pred,
    target,
    alignment_source,
    alignment_target,
    options,
    *,
    float32_alignment=False,
):
    import open3d as o3d
    from scipy.spatial import cKDTree

    scale, rotation, translation = sim3(alignment_source, alignment_target)
    aligned = scale * (pred @ rotation.T) + translation
    if float32_alignment:
        aligned = aligned.astype(np.float32)

    def cloud(points):
        value = o3d.geometry.PointCloud()
        value.points = o3d.utility.Vector3dVector(points)
        value = value.voxel_down_sample(options["voxel_m"])
        # Preserve the frozen Oxford scorer's post-voxel float32 boundary.
        if options["voxel_output_dtype"] == "float32":
            value.points = o3d.utility.Vector3dVector(
                np.asarray(value.points).astype(np.float32)
            )
        return value

    source, target_cloud = cloud(aligned), cloud(target)
    fit = o3d.pipelines.registration.registration_icp(
        source,
        target_cloud,
        options["icp_threshold_m"],
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=options["icp_iterations"]
        ),
    )
    source.transform(fit.transformation)
    a, b = np.asarray(source.points), np.asarray(target_cloud.points)
    if min(len(a), len(b)) < 3:
        raise ValueError("too few points after voxelization")
    accuracy = cKDTree(b).query(a, workers=1)[0]
    completeness = cKDTree(a).query(b, workers=1)[0]
    precision = float((accuracy < options["f1_threshold_m"]).mean())
    recall = float((completeness < options["f1_threshold_m"]).mean())
    return dict(
        cd_m=float((accuracy.mean() + completeness.mean()) / 2),
        accuracy_m=float(accuracy.mean()),
        completeness_m=float(completeness.mean()),
        precision_pct=precision * 100,
        recall_pct=recall * 100,
        f1_pct=200 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        points_pred=len(a),
        points_gt=len(b),
        sim3=dict(
            scale=float(scale),
            rotation=rotation.tolist(),
            translation=translation.tolist(),
        ),
        icp=fit.transformation.tolist(),
        icp_fitness=fit.fitness,
    )


def evaluate_dense(prediction, reference_path, protocol):
    with np.load(reference_path, allow_pickle=False) as saved:
        reference = {key: saved[key] for key in saved.files}
    poses = prediction["poses_c2w"]
    pred = predicted_points(
        prediction["depth"], prediction["intrinsics"], poses, reference, protocol
    )
    options = protocol["geometry"]
    if options["alignment"] == "camera_centers":
        target = reference["points"]
        align_pred, align_gt = poses[:, :3, 3], reference["c2w"][:, :3, 3]
    else:
        target = reference["points"][reference["mask"]]
        align = reference["alignment_mask"][reference["mask"]]
        align_pred, align_gt = pred[align], target[align]
        limit = options["alignment_max_points"]
        if limit:
            selected = np.linspace(
                0, len(align_pred) - 1, min(limit, len(align_pred)), dtype=int
            )
            align_pred, align_gt = align_pred[selected], align_gt[selected]
    result = score_clouds(
        pred,
        target,
        align_pred,
        align_gt,
        options,
        float32_alignment=protocol["dataset"] == "crocodl",
    )
    result["reference_source_id"] = str(reference["source_id"].item())
    result["shared_roi_pixels"] = int(reference["mask"].sum())
    result["f1_threshold_m"] = options["f1_threshold_m"]
    return result
