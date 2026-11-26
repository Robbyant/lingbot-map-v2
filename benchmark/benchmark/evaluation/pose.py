# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Full-trajectory Sim(3) errors and memory-bounded, all-pairs pose AUC.

AUC follows the frozen paper scorer: float32 W2C relative transforms,
Shepperd quaternions, sign-ambiguous translation, and one-degree histogram
integration of max(rotation error, translation-direction error).
"""

import numpy as np

from ..core.storage import validated_poses

THRESHOLDS = (3, 5, 15, 30)


def invert_se3(matrix):
    result = np.zeros_like(matrix)
    rotation = matrix[..., :3, :3].swapaxes(-1, -2)
    result[..., :3, :3] = rotation
    result[..., :3, 3:] = -rotation @ matrix[..., :3, 3:]
    result[..., 3, 3] = 1
    return result


def quaternion(matrix):
    """Shepperd conversion, without normalizing imperfect reference rotations."""
    r = np.asarray(matrix, dtype=float).reshape(-1, 3, 3)
    q = np.zeros((len(r), 4))
    trace = r.trace(axis1=1, axis2=2)
    positive = trace > 0
    s = 0.5 / np.sqrt(trace[positive] + 1)
    q[positive, 3] = 0.25 / s
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        q[positive, i] = (r[positive, k, j] - r[positive, j, k]) * s
    first = ~positive & (r[:, 0, 0] > r[:, 1, 1]) & (r[:, 0, 0] > r[:, 2, 2])
    second = ~positive & ~first & (r[:, 1, 1] > r[:, 2, 2])
    for mask, (i, j, k) in zip(
        (first, second, ~positive & ~first & ~second),
        ((0, 1, 2), (1, 2, 0), (2, 0, 1)),
    ):
        s = 2 * np.sqrt(1 + r[mask, i, i] - r[mask, j, j] - r[mask, k, k])
        q[mask, i] = 0.25 * s
        q[mask, 3] = (r[mask, k, j] - r[mask, j, k]) / s
        q[mask, j] = (r[mask, i, j] + r[mask, j, i]) / s
        q[mask, k] = (r[mask, i, k] + r[mask, k, i]) / s
    return q


def sim3(source, target):
    """Umeyama least-squares similarity, mapping source onto target."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Sim(3) requires corresponding [N,3] points")
    if len(source) < 3 or not np.isfinite([source, target]).all():
        raise ValueError("Sim(3) needs at least three finite correspondences")
    mx, my = source.mean(0), target.mean(0)
    x, y = source - mx, target - my
    u, d, vt = np.linalg.svd(y.T @ x / len(x))
    if (d > np.finfo(d.dtype).eps).sum() < 2:
        raise ValueError("degenerate Sim(3) alignment")
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1
    rotation = (u * sign) @ vt
    scale = (d * sign).sum() / np.square(x).sum(axis=1).mean()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid Sim(3) scale")
    return scale, rotation, my - scale * (rotation @ mx)


def _orthogonalize(poses):
    poses = np.array(poses, dtype=float, copy=True)
    u, _, vt = np.linalg.svd(poses[:, :3, :3])
    u[np.linalg.det(u @ vt) < 0, :, -1] *= -1
    poses[:, :3, :3] = u @ vt
    return poses


def trajectory_errors(gt, pred):
    gt, pred = _orthogonalize(gt), _orthogonalize(pred)
    scale, rotation, translation = sim3(pred[:, :3, 3], gt[:, :3, 3])
    pred[:, :3, 3] = scale * (pred[:, :3, 3] @ rotation.T) + translation
    pred[:, :3, :3] = rotation @ pred[:, :3, :3]
    error = invert_se3(invert_se3(gt[:-1]) @ gt[1:]) @ (
        invert_se3(pred[:-1]) @ pred[1:]
    )
    angle = np.degrees(
        np.arccos(np.clip((error[:, :3, :3].trace(axis1=1, axis2=2) - 1) / 2, -1, 1))
    )
    return {
        "ate_m": float(
            np.sqrt(np.square(pred[:, :3, 3] - gt[:, :3, 3]).sum(-1).mean())
        ),
        "rpe_translation_m": float(np.sqrt(np.square(error[:, :3, 3]).sum(-1).mean())),
        "rpe_rotation_deg": float(np.sqrt(np.square(angle).mean())),
        "sim3": {
            "scale": float(scale),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
    }


def pair_errors(gt, pred):
    loss_r = np.maximum(
        1 - np.square((quaternion(gt[:, :3, :3]) * quaternion(pred[:, :3, :3])).sum(1)),
        1e-15,
    )
    r = np.arccos(1 - 2 * loss_r) * 180 / np.pi
    a, b = gt[:, :3, 3], pred[:, :3, 3]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-15)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-15)
    loss_t = np.maximum(1 - np.square((a * b).sum(1)), 1e-15)
    t = np.arccos(np.sqrt(1 - loss_t))
    t[~np.isfinite(t)] = 1e6
    t = t * 180.0 / np.pi
    return r, np.minimum(t, np.abs(180 - t))


def auc_metrics(statistics):
    count = statistics["num_pairs"]
    return {
        f"auc_{threshold}_pct": float(
            np.cumsum(statistics["histograms"][str(threshold)]).mean() * 100 / count
        )
        for threshold in THRESHOLDS
    }


def evaluate_pose(gt, pred, *, pair_chunk=65536):
    gt, pred, valid = validated_poses(gt, pred)
    result = trajectory_errors(gt, pred)
    transforms = []
    for poses in (gt, pred):
        w2c = np.linalg.inv(poses).astype(np.float32)
        transforms.append(w2c @ invert_se3(w2c[:1])[0])
    g, p = transforms
    gi, pi = invert_se3(g), invert_se3(p)
    if type(pair_chunk) is not int or pair_chunk <= 0:
        raise ValueError("pair_chunk must be positive")
    histograms = {str(t): np.zeros(t, dtype=np.int64) for t in THRESHOLDS}
    correct = {str(t): np.zeros(2, dtype=np.int64) for t in THRESHOLDS}
    for i in range(len(gt) - 1):
        for start in range(i + 1, len(gt), pair_chunk):
            end = start + pair_chunk
            count = len(gi[start:end])
            r, t = pair_errors(
                np.repeat(g[i : i + 1], count, axis=0) @ gi[start:end],
                np.repeat(p[i : i + 1], count, axis=0) @ pi[start:end],
            )
            if not np.isfinite(r).all():
                raise ValueError("invalid reference or prediction rotation in AUC")
            error = np.maximum(r, t)
            for threshold in THRESHOLDS:
                key = str(threshold)
                histograms[key] += np.histogram(error, bins=np.arange(threshold + 1))[0]
                correct[key] += [(r < threshold).sum(), (t < threshold).sum()]
    stats = {
        "num_pairs": len(gt) * (len(gt) - 1) // 2,
        "histograms": {k: v.tolist() for k, v in histograms.items()},
    }
    result.update(auc_metrics(stats))
    result.update(
        {
            "auc_statistics": stats,
            "valid_gt_frames": int(valid.sum()),
            "input_frames": len(valid),
            "dropped_gt_indices": np.flatnonzero(~valid).tolist(),
        }
    )
    result["relative_accuracy_pct"] = {
        k: (v * 100 / stats["num_pairs"]).tolist() for k, v in correct.items()
    }
    return result
