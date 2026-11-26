# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Export one registered CroCoDL Capture session without the scantools package."""

import argparse
from pathlib import Path

import cv2
from _common import (
    camera_row,
    csv_rows,
    pose,
    source_hashes,
    undistortion,
    write_index,
    write_rgb,
)


def calibration(values):
    model, width, height, *parameters = values
    p = list(map(float, parameters))
    wh = int(width), int(height)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        expected = {"SIMPLE_PINHOLE": 3, "SIMPLE_RADIAL": 4, "RADIAL": 5}[model]
        if len(p) != expected:
            raise ValueError(f"Invalid {model} parameters")
        f, cx, cy, *radial = p
        fx = fy = f
        distortion = radial + [0.0] * (4 - len(radial))
    elif model in ("PINHOLE", "OPENCV"):
        if len(p) != (4 if model == "PINHOLE" else 8):
            raise ValueError(f"Invalid {model} parameters")
        fx, fy, cx, cy, *distortion = p
        distortion = distortion or [0.0] * 4
    else:
        raise ValueError(
            f"Unsupported camera model {model}; export a registered PINHOLE session first"
        )
    if min(fx, fy) <= 0:
        raise ValueError("Focal lengths must be positive")
    k, maps = undistortion([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], distortion, wh)
    return k, maps, wh


def export(session, trajectory, output, *, sensor=None):
    session, trajectory = (
        Path(session).resolve(strict=True),
        Path(trajectory).resolve(strict=True),
    )
    output = Path(output)
    records = csv_rows(session / "images.txt")
    sensors = {r[1] for r in records}
    if sensor is None:
        if len(sensors) != 1:
            raise ValueError("Multiple image sensors: choose --sensor explicitly")
        sensor = next(iter(sensors))
    records = sorted((r for r in records if r[1] == sensor), key=lambda r: int(r[0]))
    if not records or any(int(a[0]) >= int(b[0]) for a, b in zip(records, records[1:])):
        raise ValueError(
            "Image timestamps must be unique and increasing for the selected sensor"
        )
    cameras = {
        r[0]: r[3:] for r in csv_rows(session / "sensors.txt") if r[2] == "camera"
    }
    k, maps, wh = calibration(cameras[sensor])
    poses = {}
    for row in csv_rows(trajectory):
        key = int(row[0]), row[1]
        if key in poses:
            raise ValueError(f"Duplicate trajectory key: {key}")
        poses[key] = pose(row[2:6], row[6:9])
    rigs = []
    if (session / "rigs.txt").exists():
        rigs = [
            (r[0], pose(r[2:6], r[6:9]))
            for r in csv_rows(session / "rigs.txt")
            if r[1] == sensor
        ]
    associated = []
    for timestamp, _, filename in records:
        key = int(timestamp), sensor
        if key in poses:
            transform = poses[key]
        else:
            candidates = [
                poses[(int(timestamp), rig)] @ extrinsic
                for rig, extrinsic in rigs
                if (int(timestamp), rig) in poses
            ]
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous rig association at {timestamp}")
            transform = candidates[0] if candidates else None
        source = (session / "raw_data" / filename).resolve(strict=True)
        source.relative_to((session / "raw_data").resolve())
        associated.append((timestamp, source, transform))
    if sum(transform is not None for _, _, transform in associated) < 3:
        raise ValueError("Fewer than three images have registered ground truth")
    output.mkdir(parents=True, exist_ok=False)
    (output / "rgb").mkdir()
    rows = []
    cv2.setNumThreads(1)
    for i, (timestamp, source, transform) in enumerate(associated):
        relative = f"rgb/{timestamp}.png"
        write_rgb(source, output / relative, wh, maps)
        rows.append(camera_row(i, timestamp, relative, k, wh, transform))
    write_index(
        output,
        rows,
        dict(
            dataset="crocodl",
            frames=len(rows),
            valid_gt=sum(r["has_gt"] for r in rows),
            pose_convention="camera_to_world = rig_to_world @ camera_to_rig",
            association="exact timestamp and sensor/rig ID; no interpolation",
            undistortion_alpha=0,
            sources=source_hashes(
                {
                    "trajectory": trajectory,
                    "sensors": session / "sensors.txt",
                    "images": session / "images.txt",
                    **(
                        {"rigs": session / "rigs.txt"}
                        if (session / "rigs.txt").exists()
                        else {}
                    ),
                }
            ),
        ),
    )
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session", required=True, help="Capture sessions/<session> directory"
    )
    parser.add_argument(
        "--trajectory",
        required=True,
        help="Registered sensor/rig-to-world trajectory, not local phone tracking",
    )
    parser.add_argument(
        "--output", required=True, help="New processed <site>__<session> directory"
    )
    parser.add_argument(
        "--sensor", help="Image sensor ID, required for multisensor sessions"
    )
    args = parser.parse_args()
    print(
        f"Exported {export(args.session, args.trajectory, args.output, sensor=args.sensor)} frames"
    )
