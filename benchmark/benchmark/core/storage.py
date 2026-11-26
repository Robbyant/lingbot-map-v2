# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Small shared helpers for immutable benchmark plans and result receipts."""

import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def resolve(base, path):
    return (Path(base) / path).expanduser().resolve(strict=True)


def load_array(path, key=None):
    """Tensor-only data: never accept pickle/object arrays."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is None:
                if len(archive.files) != 1:
                    raise ValueError(f"{path}: specify an array key")
                key = archive.files[0]
            return archive[key]
    if key is not None:
        raise ValueError("array key is supported only for .npz files")
    return np.load(path, allow_pickle=False, mmap_mode="r")


def validated_poses(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    if gt.shape != pred.shape or gt.ndim != 3 or gt.shape[1:] != (4, 4):
        raise ValueError("matching [N,4,4] C2W trajectories are required")
    if not np.isfinite(pred).all():
        raise ValueError(
            "nonfinite predictions must fail, not shrink the evaluation set"
        )
    valid = np.isfinite(gt).all(axis=(1, 2))
    if valid.sum() < 3:
        raise ValueError("at least three valid GT poses are required")
    return gt[valid], pred[valid], valid


def source_identity():
    """Hash actual source contents, including local edits, without relying on Git."""
    import lingbot_map2

    package = Path(lingbot_map2.__file__).parent
    entries = {}
    for root in (package, Path(__file__).resolve().parents[2]):
        for path in sorted(root.rglob("*.py")):
            entries[f"{root.name}/{path.relative_to(root)}"] = sha256(path)
    return identity(entries)
