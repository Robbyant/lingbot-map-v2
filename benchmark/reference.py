# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Prepare fixed dense references independently of all model predictions.

Camera bundles contain selected raw_ids, C2W, raw-camera K and raw_wh.
The shared footprint is an explicit scoring-grid Boolean array, frozen before
inference. It is never inferred from a model's confidence or predicted depth.
"""

import argparse
from pathlib import Path

import numpy as np

from .benchmark.core.storage import read_json, resolve, sha256, identity, write_json
from .datasets.protocols import PROTOCOLS, selection
from .benchmark.evaluation.points import validate_reference


def pixel_transform(raw_wh, output_wh, *, cover=False):
    w, h = raw_wh
    tw, th = output_wh
    if cover:
        scale = max(tw / w, th / h)
        rw, rh = round(w * scale), round(h * scale)
        left, top = (rw - tw) // 2, (rh - th) // 2
    else:
        rw, rh, left, top = tw, th, 0, 0
    sx, sy = rw / w, rh / h
    return np.array(
        [[sx, 0, sx * 0.5 - 0.5 - left], [0, sy, sy * 0.5 - 0.5 - top], [0, 0, 1.0]]
    )


def scan_visibility(xyz, poses, intrinsics, footprint, near=0.1, far=20.0):
    """Nearest positive-z scan point at each pixel; retain the visible union."""
    h, w = footprint.shape
    selected = np.zeros(len(xyz), dtype=bool)
    masks = []
    for pose, k in zip(poses, intrinsics):
        camera = (xyz - pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        ids = np.flatnonzero((z > near) & (z < far))
        uv = camera[ids] @ k.T
        uv = np.floor(uv[:, :2] / uv[:, 2:] + 0.5).astype(int)
        good = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids, uv = ids[good], uv[good]
        flat = uv[:, 1] * w + uv[:, 0]
        order = np.lexsort((z[ids], flat))
        _, first = np.unique(flat[order], return_index=True)
        take = order[first]
        take = take[footprint.ravel()[flat[take]]]
        selected[ids[take]] = True
        mask = np.zeros(h * w, dtype=bool)
        mask[flat[take]] = True
        masks.append(mask.reshape(h, w))
    return np.asarray(masks), np.flatnonzero(selected)


def prepare_reference(spec_path, output):
    spec_path = Path(spec_path).resolve(strict=True)
    spec = read_json(spec_path)
    protocol = PROTOCOLS[spec["protocol"]]
    if protocol["kind"] != "dense":
        raise ValueError("reference preparation requires a dense protocol")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    expected_ids, _ = selection(protocol, spec["raw_count"])
    camera_path = resolve(spec_path.parent, spec["cameras"])
    with np.load(camera_path, allow_pickle=False) as data:
        ids, poses, raw_k, raw_wh = (data[k] for k in ("raw_ids", "c2w", "K", "raw_wh"))
    np.testing.assert_array_equal(ids, expected_ids)
    if (
        poses.shape != (len(ids), 4, 4)
        or raw_k.shape != (len(ids), 3, 3)
        or raw_wh.shape != (2,)
    ):
        raise ValueError(
            "invalid camera bundle; varying raw image sizes must be split during data preparation"
        )
    if not np.isfinite(poses).all() or not np.isfinite(raw_k).all():
        raise ValueError("dense reference cameras must be finite")
    w, h = protocol["geometry"]["grid_wh"]
    footprint_path = resolve(spec_path.parent, spec["footprint"])
    footprint = np.load(footprint_path, allow_pickle=False)
    if footprint.shape != (h, w) or footprint.dtype != bool:
        raise ValueError(
            "footprint must be a Boolean [grid_height,grid_width] shared-FOV mask"
        )
    ours = pixel_transform(raw_wh, protocol["input_wh"], cover=True)
    grid = pixel_transform(protocol["input_wh"], [w, h]) @ ours
    k = grid[None] @ raw_k
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack([xx, yy, np.ones_like(xx)], -1)
    native = pixels @ np.linalg.inv(grid).T @ ours.T
    xy = np.floor(native[..., :2] + 0.5).astype(np.int64)
    sources = dict(
        cameras=sha256(camera_path),
        footprint=sha256(footprint_path),
        spec=sha256(spec_path),
    )
    dataset = protocol["dataset"]
    if dataset == "euroc":
        import open3d as o3d

        scan = resolve(spec_path.parent, spec["scan"])
        xyz = np.asarray(o3d.io.read_point_cloud(str(scan)).points)
        if len(xyz) < 3 or not np.isfinite(xyz).all():
            raise ValueError("empty or nonfinite scanner cloud")
        mask, indices = scan_visibility(xyz, poses, k, footprint)
        points, alignment_mask = xyz[indices], mask.copy()
        sources["scan"] = sha256(scan)
    else:
        if dataset == "crocodl":
            import open3d as o3d

            mesh_path = resolve(spec_path.parent, spec["mesh"])
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            raycaster = o3d.t.geometry.RaycastingScene(nthreads=1)
            raycaster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            depths = []
            for pose, intrinsics in zip(poses, k):
                directions = (pixels @ np.linalg.inv(intrinsics).T) @ pose[:3, :3].T
                rays = np.concatenate(
                    [np.broadcast_to(pose[:3, 3], directions.shape), directions], -1
                ).astype(np.float32)
                depths.append(
                    raycaster.cast_rays(o3d.core.Tensor(rays), nthreads=1)[
                        "t_hit"
                    ].numpy()
                )
            depths = np.asarray(depths)
            sources["mesh"] = sha256(mesh_path)
            mask = np.isfinite(depths) & (depths > 0.1) & (depths < 50) & footprint
            alignment_mask = mask.copy()
        else:
            # Oxford consumes the published/projected GT grid, not a rerendered scan.
            depth_path = resolve(spec_path.parent, spec["depth_grid"])
            depths = np.load(depth_path, allow_pickle=False)
            grid_k_path = resolve(spec_path.parent, spec["grid_intrinsics"])
            k = np.load(grid_k_path, allow_pickle=False)
            if depths.shape != (len(ids), h, w) or k.shape != (len(ids), 3, 3):
                raise ValueError(
                    "Oxford needs associated 518x388 metric depths and grid intrinsics"
                )
            # Frozen common ROI rows [50,338), with a separate GT-only alignment range.
            required = np.zeros((h, w), bool)
            required[50:338] = True
            if not np.array_equal(footprint, required):
                raise ValueError(
                    "Oxford common ROI must be rows [50,338) on the 518x388 GT grid"
                )
            mask = np.isfinite(depths) & (depths > 0.1) & (depths < 200) & footprint
            alignment_mask = mask & (depths < 80)
            sources.update(
                depth_grid=sha256(depth_path), grid_intrinsics=sha256(grid_k_path)
            )
        points = np.zeros((*mask.shape, 3), np.float32)
        for i, (pose, intrinsics) in enumerate(zip(poses, k)):
            ray = (pixels @ np.linalg.inv(intrinsics).T) @ pose[:3, :3].T
            valid = mask[i]
            points[i][valid] = ray[valid] * depths[i][valid, None] + pose[:3, 3]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Opening exclusively prevents accidental overwrite of a frozen reference.
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            raw_ids=ids,
            input_wh=protocol["input_wh"],
            mask=mask,
            alignment_mask=alignment_mask,
            points=points,
            c2w=poses,
            pixel_xy=xy,
            source_id=identity(sources),
        )
    validate_reference(output, expected_ids, protocol)
    write_json(
        output.with_suffix(".provenance.json"),
        dict(protocol=protocol, sources=sources, reference_sha256=sha256(output)),
    )
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare_reference(args.spec, args.output))


if __name__ == "__main__":
    main()
