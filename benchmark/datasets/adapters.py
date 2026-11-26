# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Convert common on-disk camera formats to explicit raw-indexed manifests.

No guessed time offsets, camera extrinsics, interpolation or GT-based input
filtering. Scene datasets may instead supply the documented manifest directly.
"""

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

import numpy as np

from ..benchmark.core.storage import write_json
from .protocols import PROTOCOLS, get_protocol


def tum_by_frame(path, count):
    """Aligned TUM C2W: first column indexes the ordered RGB sequence."""
    from scipy.spatial.transform import Rotation

    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] != 8:
        raise ValueError("expected frame_id tx ty tz qx qy qz qw")
    ids = rows[:, 0].astype(np.int64)
    if (
        not np.array_equal(ids, rows[:, 0])
        or len(set(ids)) != len(ids)
        or (ids < 0).any()
        or (ids >= count).any()
    ):
        raise ValueError(
            "TUM timestamps must already be associated to unique RGB sequence indices"
        )
    poses = np.full((count, 4, 4), np.nan)
    valid = np.isfinite(rows[:, 1:]).all(1) & (np.linalg.norm(rows[:, 4:], axis=1) > 0)
    selected = ids[valid]
    poses[selected] = np.eye(4)
    poses[selected, :3, :3] = Rotation.from_quat(rows[valid, 4:]).as_matrix()
    poses[selected, :3, 3] = rows[valid, 1:4]
    return poses


def _images(folder, suffix, first=0, *, consecutive=True):
    paths = sorted(folder.glob(suffix), key=lambda p: int(p.stem))
    indices = [int(p.stem) for p in paths]
    if not paths:
        raise ValueError(f"No images in {folder}")
    if indices[0] < 0 or len(set(indices)) != len(indices):
        raise ValueError(f"Negative or duplicate image indices in {folder}")
    if consecutive and indices != list(range(first, first + len(paths))):
        raise ValueError(f"Missing or nonconsecutive image indices in {folder}")
    return paths


def camera_rows(root, camera_index):
    """Read complete camera records in exact numeric timestamp order."""
    path = root / camera_index
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty camera index: {path}")

    # Sort complete records together; never sort RGB and poses independently.
    def times(row):
        try:
            value = Decimal(str(row.get("timestamp", row.get("timestamp_us"))))
        except InvalidOperation as error:
            raise ValueError(f"Invalid timestamp in {path}") from error
        if not value.is_finite():
            raise ValueError(f"Nonfinite timestamp in {path}")
        return value

    rows.sort(key=times)
    stamps = [times(r) for r in rows]
    if not all(t.is_finite() for t in stamps) or any(
        a >= b for a, b in zip(stamps, stamps[1:])
    ):
        raise ValueError(f"Camera timestamps must be finite and unique: {path}")
    return rows


def _processed(root, camera_index):
    path = root / camera_index
    rows = camera_rows(root, camera_index)
    images, poses = [], []
    for row in rows:
        relative = Path(row.get("image", row.get("rgb", "")))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected scene-relative RGB paths: {path}")
        if row.get("camera_model", "PINHOLE") != "PINHOLE":
            raise ValueError("Processed RGB must already be undistorted PINHOLE images")
        images.append(root / relative)
        pose = row.get("c2w")
        valid = row.get("gt_valid", row.get("has_gt", pose is not None))
        if valid and (pose is None or not np.isfinite(pose).all()):
            raise ValueError(f"Declared valid GT pose is missing/nonfinite: {path}")
        poses.append(pose if valid else np.full((4, 4), np.nan))
    return dict(images=[str(p) for p in images]), np.asarray(poses, dtype=float)


def read_scene(protocol_name, root, scene, *, camera_index="camera/frames.jsonl"):
    """Read one prepared dataset scene without importing another model/framework."""
    protocol = PROTOCOLS[protocol_name]
    dataset = protocol["dataset"]
    root = Path(root).resolve(strict=True)
    folder = root / scene
    if dataset == "kitti":
        images = _images(root / "sequences" / scene / "image_2", "*.png")
        values = np.loadtxt(root / "poses" / f"{scene}.txt").reshape(-1, 3, 4)
        poses = np.tile(np.eye(4), (len(values), 1, 1))
        poses[:, :3] = values
        media = dict(images=[str(p) for p in images])
    elif dataset == "vbr":
        # LoGeR keeps source filenames; aligned GT indexes the sorted RGB list.
        images = _images(
            root / f"{scene}_processed_aligned" / "rgb", "*.png", consecutive=False
        )
        poses = tum_by_frame(root / "processed_gt" / f"{scene}_gt.txt", len(images))
        media = dict(images=[str(p) for p in images])
    elif dataset == "oxford":
        images = _images(folder / "images", "*.png")
        poses = np.loadtxt(folder / "poses_c2w.txt").reshape(-1, 4, 4)
        media = dict(images=[str(p) for p in images])
    elif dataset == "tat":
        images = _images(folder, "*.jpg", consecutive=False)
        image_ids = [int(p.stem) - 1 for p in images]
        if image_ids[0] < 0:
            raise ValueError("TAT image filenames must be positive one-based indices")
        lines = [
            line.strip()
            for line in (folder / f"{scene}_COLMAP_SfM.log").read_text().splitlines()
            if line.strip()
        ]
        if not lines or len(lines) % 5:
            raise ValueError("TAT camera logs require five lines per C2W entry")
        matrices = {}
        for i in range(len(lines) // 5):
            header = lines[5 * i].split()
            if len(header) != 3 or int(header[0]) != int(header[1]):
                raise ValueError("TAT log entries require matching camera indices")
            camera_id = int(header[0])
            if camera_id < 0 or camera_id in matrices:
                raise ValueError(
                    "TAT log camera indices must be unique and nonnegative"
                )
            matrices[camera_id] = [
                [float(x) for x in line.split()]
                for line in lines[5 * i + 1 : 5 * i + 5]
            ]
        if not set(image_ids) <= set(matrices):
            raise ValueError(
                "TAT log is missing camera indices specified by image filenames"
            )
        poses = np.asarray([matrices[i] for i in image_ids])
        media = dict(images=[str(p) for p in images])
    elif dataset in ("crocodl", "euroc"):
        media, poses = _processed(folder, camera_index)
    else:
        from .princeton365 import read_princeton365

        video, poses = read_princeton365(root, scene)
        media = dict(video=str(video.resolve(strict=True)))
    count = len(poses)
    if poses.shape != (count, 4, 4) or count < 8:
        raise ValueError(f"{scene}: expected raw-indexed [N,4,4] C2W")
    if "images" in media:
        paths = media["images"]
        if len(paths) != count or len(set(paths)) != count:
            raise ValueError(f"{scene}: duplicate images or mismatched RGB/GT counts")
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    if "video" in media:
        import cv2

        capture = cv2.VideoCapture(media["video"])
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open RGB video: {media['video']}")
            reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if reported > 0 and reported != count:
                raise ValueError(f"Video reports {reported} frames but GT has {count}")
        finally:
            capture.release()
    valid = np.isfinite(poses).all((1, 2))
    if valid.sum() < 3 or not np.allclose(poses[valid, 3], [0, 0, 0, 1]):
        raise ValueError(f"{scene}: insufficient or malformed camera poses")
    return dict(name=scene, raw_count=count, **media), poses


def inspect_dataset(
    protocol_name,
    root,
    *,
    scenes=None,
    reference_root=None,
    camera_index="camera/frames.jsonl",
):
    protocol = PROTOCOLS[protocol_name]
    root = Path(root).resolve(strict=True)
    selected = protocol["expected_scenes"] if scenes is None else scenes
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - set(protocol["expected_scenes"])
    ):
        raise ValueError("Select unique scene names from the protocol's scene list")
    rows, errors = [], []
    for scene in selected:
        try:
            row, poses = read_scene(
                protocol_name, root, scene, camera_index=camera_index
            )
            if row["raw_count"] != protocol["expected_raw_counts"][scene]:
                raise ValueError(
                    f"raw_count={row['raw_count']}; paper split requires {protocol['expected_raw_counts'][scene]}"
                )
            if protocol["kind"] == "dense":
                if reference_root is None:
                    raise ValueError(
                        "reference_root is required for shared-ROI dense evaluation"
                    )
                reference = (
                    Path(reference_root).resolve(strict=True) / f"{scene}.reference.npz"
                )
                from ..benchmark.evaluation.points import validate_reference
                from .protocols import selection

                ids, _ = selection(protocol, row["raw_count"])
                validate_reference(reference, ids, protocol)
                with np.load(reference, allow_pickle=False) as data:
                    if not np.allclose(poses[ids], data["c2w"], atol=1e-6, rtol=0):
                        raise ValueError(
                            "camera revision/gauge differs from the dense reference"
                        )
                row["reference"] = str(reference)
            rows.append((row, poses))
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            errors.append(f"{scene}: {error}")
    if errors:
        raise ValueError("Dataset preflight failed:\n" + "\n".join(errors))
    return rows


def prepare_native(
    protocol_name,
    root,
    output,
    *,
    scenes=None,
    reference_root=None,
    camera_index="camera/frames.jsonl",
    input_scope=None,
):
    protocol = get_protocol(protocol_name, input_scope)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = inspect_dataset(
        protocol_name,
        root,
        scenes=scenes,
        reference_root=reference_root,
        camera_index=camera_index,
    )
    output.mkdir(parents=True)
    entries = []
    for row, poses in rows:
        target = output / f"{row['name']}.gt.npy"
        np.save(target, poses)
        entries.append(dict(row, gt_poses=target.name))
    spec = dict(
        protocol=protocol_name, input_scope=protocol["input_scope"], scenes=entries
    )
    write_json(output / "dataset.json", spec)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=list(PROTOCOLS),
        required=True,
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--reference-root")
    parser.add_argument("--camera-index", default="camera/frames.jsonl")
    parser.add_argument(
        "--input-scope", choices=["full_stream", "keyframes", "selected_views"]
    )
    args = parser.parse_args()
    prepare_native(
        args.protocol,
        args.root,
        args.output,
        scenes=args.scene,
        reference_root=args.reference_root,
        camera_index=args.camera_index,
        input_scope=args.input_scope,
    )
