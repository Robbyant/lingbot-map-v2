# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Export an official EuRoC cam0 stream to undistorted RGB and camera C2W."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from _common import (
    camera_row,
    csv_rows,
    pose,
    rigid,
    source_hashes,
    undistortion,
    write_index,
    write_rgb,
)
from scipy.spatial.transform import Rotation, Slerp


def associated_poses(stamps, state_rows, body_from_camera):
    times = np.asarray([int(r[0]) for r in state_rows], dtype=np.int64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Ground-truth timestamps must be strictly increasing")
    body = np.stack([pose(r[4:8], r[1:4]) for r in state_rows])
    # Subtract the integer origin before converting nanoseconds to float seconds.
    t = (times - times[0]).astype(np.float64) * 1e-9
    query = (np.asarray(stamps, dtype=np.int64) - times[0]).astype(np.float64) * 1e-9
    valid = (query >= t[0]) & (query <= t[-1])
    c2w = np.full((len(query), 4, 4), np.nan)
    if valid.any():
        interpolated = np.tile(np.eye(4), (int(valid.sum()), 1, 1))
        interpolated[:, :3, :3] = Slerp(t, Rotation.from_matrix(body[:, :3, :3]))(
            query[valid]
        ).as_matrix()
        for axis in range(3):
            interpolated[:, axis, 3] = np.interp(query[valid], t, body[:, axis, 3])
        c2w[valid] = interpolated @ rigid(body_from_camera)
    return c2w


def export(sequence, output, *, ground_truth=None):
    sequence, output = Path(sequence).resolve(strict=True), Path(output)
    mav = sequence if sequence.name == "mav0" else sequence / "mav0"
    cam = mav / "cam0"
    calibration = yaml.safe_load((cam / "sensor.yaml").read_text())
    if (
        calibration["camera_model"] != "pinhole"
        or calibration["distortion_model"] != "radial-tangential"
    ):
        raise ValueError("Expected EuRoC pinhole/radial-tangential calibration")
    fx, fy, cx, cy = calibration["intrinsics"]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    wh = calibration["resolution"]
    k, maps = undistortion(k, calibration["distortion_coefficients"], wh)
    tbc = rigid(np.asarray(calibration["T_BS"]["data"]).reshape(4, 4))
    records = sorted(csv_rows(cam / "data.csv"), key=lambda r: int(r[0]))
    stamps = [int(r[0]) for r in records]
    if not stamps or any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("RGB timestamps must be unique and increasing")
    truth = (
        Path(ground_truth)
        if ground_truth
        else mav / "state_groundtruth_estimate0/data.csv"
    )
    poses = associated_poses(stamps, csv_rows(truth), tbc)
    if np.isfinite(poses).all((1, 2)).sum() < 3:
        raise ValueError("Fewer than three RGB frames have associated ground truth")
    output.mkdir(parents=True, exist_ok=False)
    (output / "rgb").mkdir()
    rows = []
    cv2.setNumThreads(1)
    for i, (timestamp, filename) in enumerate(records):
        source = (cam / "data" / filename).resolve(strict=True)
        source.relative_to((cam / "data").resolve())
        relative = f"rgb/{timestamp}.png"
        write_rgb(source, output / relative, wh, maps)
        rows.append(
            camera_row(
                i,
                timestamp,
                relative,
                k,
                wh,
                poses[i] if np.isfinite(poses[i]).all() else None,
            )
        )
    write_index(
        output,
        rows,
        dict(
            dataset="euroc",
            frames=len(rows),
            valid_gt=int(np.isfinite(poses).all((1, 2)).sum()),
            pose_convention="T_WC = T_WB @ T_BS(cam0)",
            association="linear translation and SLERP; no extrapolation",
            undistortion_alpha=0,
            sources=source_hashes(
                {
                    "calibration": cam / "sensor.yaml",
                    "images": cam / "data.csv",
                    "ground_truth": truth,
                }
            ),
        ),
    )
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", required=True, help="Official sequence directory containing mav0"
    )
    parser.add_argument("--output", required=True, help="New processed scene directory")
    parser.add_argument(
        "--ground-truth", help="Override state_groundtruth_estimate0/data.csv"
    )
    args = parser.parse_args()
    print(
        f"Exported {export(args.sequence, args.output, ground_truth=args.ground_truth)} frames"
    )
