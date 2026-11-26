# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Build camera bundles, shared-FOV masks and dense reference specifications."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.datasets.adapters import camera_rows, read_scene
from benchmark.datasets.protocols import PROTOCOLS, selection
from benchmark.reference import pixel_transform, prepare_reference


def common_footprint(protocol, raw_wh):
    """Intersect calibrated image canvases, independently of all predictions."""
    w, h = protocol["geometry"]["grid_wh"]
    if protocol["dataset"] == "oxford":
        mask = np.zeros((h, w), bool)
        mask[50:338] = True
        return mask
    ours = pixel_transform(raw_wh, protocol["input_wh"], cover=True)
    grid = pixel_transform(protocol["input_wh"], (w, h)) @ ours
    yy, xx = np.mgrid[:h, :w]
    raw = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(grid).T
    # Width-locked canvas with centered vertical crop/padding.
    resized_h = round(raw_wh[1] * 504 / raw_wh[0])
    crop_top = round((resized_h - 280) / 2) if resized_h > 280 else 0
    pad_top = (280 - resized_h) // 2 if resized_h < 280 else 0
    transform = pixel_transform(raw_wh, (504, resized_h))
    transform[1, 2] += pad_top - crop_top
    uv = np.floor((raw @ transform.T)[..., :2] + 0.5).astype(int)
    mask = (uv[..., 0] >= 0) & (uv[..., 0] < 504)
    mask &= (uv[..., 1] >= pad_top) & (uv[..., 1] < pad_top + min(resized_h, 280))
    if protocol["dataset"] == "euroc":
        uv = np.floor(
            (raw @ pixel_transform(raw_wh, (518, 336), cover=True).T)[..., :2] + 0.5
        ).astype(int)
        mask &= (
            (uv[..., 0] >= 0)
            & (uv[..., 0] < 518)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < 336)
        )
    if not mask.any():
        raise ValueError("Empty common camera footprint")
    return mask


def prepare(
    protocol_name,
    root,
    scene,
    output,
    *,
    geometry=None,
    camera_index="camera/frames.jsonl",
    build_reference=False,
    reference_root=None,
):
    if reference_root is not None and not build_reference:
        raise ValueError("--reference-root requires --build-reference")
    protocol = PROTOCOLS[protocol_name]
    root = Path(root).resolve(strict=True)
    output = Path(output)
    reference = (
        Path(reference_root).expanduser() if reference_root is not None else output
    ) / f"{scene}.reference.npz"
    if build_reference and reference.exists():
        raise FileExistsError(f"Reference already exists: {reference}")
    row, poses = read_scene(protocol_name, root, scene, camera_index=camera_index)
    if (
        scene not in protocol["expected_scenes"]
        or row["raw_count"] != protocol["expected_raw_counts"][scene]
    ):
        raise ValueError("Scene/count differs from the prescribed split")
    ids, _ = selection(protocol, row["raw_count"])
    if not np.isfinite(poses[ids]).all():
        raise ValueError("Selected dense GT cameras must all be valid")
    folder = root / scene
    if protocol["dataset"] == "oxford":
        calibration = np.loadtxt(folder / "intrinsics.txt").reshape(-1)
        if calibration.shape != (6,):
            raise ValueError(
                "Oxford intrinsics.txt must contain fx fy cx cy width height"
            )
        fx, fy, cx, cy, width, height = calibration
        if width != int(width) or height != int(height):
            raise ValueError("Image dimensions must be integers")
        raw_wh = (int(width), int(height))
        intrinsics = np.tile([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], (len(ids), 1, 1))
    else:
        records = camera_rows(folder, camera_index)
        sizes = {(r["width"], r["height"]) for r in records}
        if len(sizes) != 1:
            raise ValueError("Image dimensions must be constant within a dense scene")
        raw_wh = tuple(map(int, sizes.pop()))
        intrinsics = np.asarray([records[i]["K"] for i in ids], dtype=float)
        if geometry is None:
            raise ValueError("Provide the registered mesh/scan with --geometry")
        geometry = Path(geometry).resolve(strict=True)
    if (
        intrinsics.shape != (len(ids), 3, 3)
        or not np.isfinite(intrinsics).all()
        or min(raw_wh) <= 0
    ):
        raise ValueError("Invalid raw camera calibration")
    if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise ValueError("Focal lengths must be positive")
    footprint = common_footprint(protocol, raw_wh)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(
        output / "cameras.npz", raw_ids=ids, c2w=poses[ids], K=intrinsics, raw_wh=raw_wh
    )
    np.save(output / "common-footprint.npy", footprint)
    spec = dict(
        protocol=protocol_name,
        raw_count=row["raw_count"],
        cameras="cameras.npz",
        footprint="common-footprint.npy",
    )
    if protocol["dataset"] == "oxford":
        w, h = protocol["geometry"]["grid_wh"]
        depths = np.lib.format.open_memmap(
            output / "depth-grid.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(ids), h, w),
        )
        for j, i in enumerate(ids):
            source = folder / "depth" / (Path(row["images"][i]).stem + ".npy")
            depth = np.load(source, allow_pickle=False)
            if depth.shape != raw_wh[::-1]:
                raise ValueError(
                    f"Metric depth shape differs from its camera: {source}"
                )
            depths[j] = cv2.resize(
                depth.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
            )
        depths.flush()
        del depths
        grid_k = intrinsics.copy()
        grid_k[:, 0] *= w / raw_wh[0]
        grid_k[:, 1] *= h / raw_wh[1]
        np.save(output / "grid-intrinsics.npy", grid_k)
        spec.update(depth_grid="depth-grid.npy", grid_intrinsics="grid-intrinsics.npy")
    else:
        spec["mesh" if protocol["dataset"] == "crocodl" else "scan"] = str(geometry)
    spec_path = output / "reference-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    if build_reference:
        prepare_reference(spec_path, reference)
    return reference if build_reference else spec_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        required=True,
        choices=[k for k, v in PROTOCOLS.items() if v["kind"] == "dense"],
    )
    parser.add_argument(
        "--root", required=True, help="Prepared RGB/camera dataset root"
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--output", required=True, help="New per-scene bundle directory"
    )
    parser.add_argument(
        "--geometry", help="Registered NavVis mesh (CroCoDL) or Leica scan (EuRoC)"
    )
    parser.add_argument("--camera-index", default="camera/frames.jsonl")
    parser.add_argument(
        "--build-reference",
        action="store_true",
        help="Also run CPU reference construction",
    )
    parser.add_argument(
        "--reference-root",
        help="Write <scene>.reference.npz here with --build-reference; defaults to --output",
    )
    args = parser.parse_args()
    print(
        prepare(
            args.protocol,
            args.root,
            args.scene,
            args.output,
            geometry=args.geometry,
            camera_index=args.camera_index,
            build_reference=args.build_reference,
            reference_root=args.reference_root,
        )
    )
