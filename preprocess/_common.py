# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Small utilities for exporting calibrated, timestamp-associated RGB streams."""

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def source_hashes(paths):
    result = {}
    for label, path in paths.items():
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[label] = digest.hexdigest()
    return result


def csv_rows(path):
    with Path(path).open() as stream:
        return [
            [v.strip() for v in row]
            for row in csv.reader(
                line
                for line in stream
                if line.strip() and not line.lstrip().startswith("#")
            )
        ]


def pose(qwxyz, translation):
    q = np.asarray(qwxyz, dtype=float)
    t = np.asarray(translation, dtype=float)
    if (
        q.shape != (4,)
        or t.shape != (3,)
        or not np.isfinite(np.r_[q, t]).all()
        or np.linalg.norm(q) == 0
    ):
        raise ValueError("Expected a finite quaternion and translation")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = t
    return result


def rigid(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Expected a rigid homogeneous transform")
    return matrix


def undistortion(k, distortion, wh):
    """Keep the original image size; use the alpha=0 pinhole canvas."""
    k, distortion = np.asarray(k, float), np.asarray(distortion, float)
    wh = tuple(map(int, wh))
    if not np.isfinite(k).all() or not np.isfinite(distortion).all() or min(wh) <= 0:
        raise ValueError("Invalid camera calibration")
    if not np.any(distortion):
        return k, None
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, distortion, wh, 0, wh)
    maps = cv2.initUndistortRectifyMap(k, distortion, None, new_k, wh, cv2.CV_32FC1)
    return new_k, maps


def write_rgb(source, destination, wh, maps):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.shape[1::-1] != tuple(wh):
        raise ValueError(f"RGB dimensions disagree with calibration: {source}")
    if maps is not None:
        image = cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Cannot write {destination}")


def camera_row(index, timestamp, image, k, wh, c2w):
    return dict(
        image_id=index,
        timestamp=str(timestamp),
        image=image,
        camera_model="PINHOLE",
        K=np.asarray(k).tolist(),
        width=int(wh[0]),
        height=int(wh[1]),
        c2w=None if c2w is None else rigid(c2w).tolist(),
        has_gt=c2w is not None,
    )


def write_index(output, rows, metadata):
    camera = Path(output) / "camera"
    camera.mkdir()
    with (camera / "frames.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (Path(output) / "preparation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
