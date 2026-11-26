# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Freeze explicit RGB/GT associations into a hash-checked evaluation plan."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "benchmark"

import argparse
import re

import numpy as np

from .benchmark.core.storage import (
    read_json, write_json, resolve, sha256, identity, load_array,
)
from .datasets.protocols import get_protocol, selection


def prepare(spec_path, output):
    spec_path, output = Path(spec_path).resolve(strict=True), Path(output)
    if output.exists():
        raise FileExistsError(output)
    spec = read_json(spec_path)
    protocol = get_protocol(spec["protocol"], spec.get("input_scope"))
    diagnostic = spec.get("diagnostic", False)
    if type(diagnostic) is not bool:
        raise ValueError("diagnostic must be a Boolean")
    rows, names = [], set()
    # Validate everything before creating an immutable output directory.
    for scene in spec["scenes"]:
        name = scene["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise ValueError("scene names must be unique safe path components")
        if name not in protocol["expected_scenes"]:
            raise ValueError(f"unexpected scene for {protocol['name']}: {name}")
        names.add(name)
        count = scene["raw_count"]
        if not diagnostic and count != protocol["expected_raw_counts"][name]:
            raise ValueError(
                f"{name}: raw_count differs from the frozen paper split; use diagnostic=true for a labeled smoke test"
            )
        ids, persistent = selection(protocol, count)
        gt_path = resolve(spec_path.parent, scene["gt_poses"])
        gt = load_array(gt_path, scene.get("gt_key"))
        selected = "source_indices" in scene
        if selected and scene["source_indices"] != ids:
            raise ValueError(
                "selected source indices must exactly match the fixed protocol"
            )
        expected_gt = len(ids) if selected else count
        if gt.shape != (expected_gt, 4, 4):
            raise ValueError(f"{name}: GT must be associated [N,4,4] C2W")
        gt = gt if selected else gt[ids]
        if np.isfinite(gt).all((1, 2)).sum() < 3:
            raise ValueError("at least three selected GT poses must be valid")
        # Invalid GT stays in the plan; it must never filter inference inputs.
        row = dict(
            name=name,
            raw_count=count,
            raw_ids=ids,
            persistent=persistent,
            gt_source_sha256=sha256(gt_path),
            gt_selected=np.asarray(gt),
        )
        if ("images" in scene) == ("video" in scene):
            raise ValueError("provide exactly one ordered image list or video")
        if "images" in scene:
            paths = scene["images"]
            if len(paths) != expected_gt or len(set(paths)) != expected_gt:
                raise ValueError("images must list each prescribed RGB exactly once")
            selected_paths = [
                resolve(spec_path.parent, p)
                for p in (paths if selected else [paths[i] for i in ids])
            ]
            if len(set(selected_paths)) != len(selected_paths):
                raise ValueError("RGB path aliases must not duplicate input frames")
            row["images"] = [str(p) for p in selected_paths]
            row["image_sha256"] = [sha256(p) for p in selected_paths]
        else:
            video = resolve(spec_path.parent, scene["video"])
            row.update(video=str(video), video_sha256=sha256(video))
        if protocol["kind"] == "dense":
            reference = resolve(spec_path.parent, scene["reference"])
            from .benchmark.evaluation.points import validate_reference

            validate_reference(reference, ids, protocol)
            row.update(reference=str(reference), reference_sha256=sha256(reference))
        rows.append(row)
    if not rows:
        raise ValueError("no scenes requested")
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        gt = row.pop("gt_selected")
        target = output / f"{row['name']}.gt.npy"
        np.save(target, gt)
        row.update(gt_file=target.name, gt_sha256=sha256(target))
    plan = dict(
        schema=1,
        protocol=protocol,
        diagnostic=diagnostic,
        scenes=rows,
        spec_sha256=sha256(spec_path),
    )
    plan["plan_id"] = identity(plan)
    write_json(output / "plan.json", plan)
    return plan


def load_plan(path):
    path = Path(path).resolve(strict=True)
    plan = read_json(path)
    body = {k: v for k, v in plan.items() if k != "plan_id"}
    if identity(body) != plan["plan_id"]:
        raise ValueError("plan integrity check failed")
    if plan["protocol"] != get_protocol(
        plan["protocol"]["name"], plan["protocol"].get("input_scope")
    ):
        raise ValueError("plan protocol differs from this benchmark version")
    for row in plan["scenes"]:
        ids, persistent = selection(plan["protocol"], row["raw_count"])
        if row["raw_ids"] != ids or row["persistent"] != persistent:
            raise ValueError("plan frame selection differs from its protocol")
        if sha256(path.parent / row["gt_file"]) != row["gt_sha256"]:
            raise ValueError("GT changed after planning")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec")
    parser.add_argument("--output")
    parser.add_argument("--config", help="base YAML under benchmark/configs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved settings without data access or CUDA",
    )
    parser.add_argument("--scene", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate data layout without writing outputs",
    )
    args = parser.parse_args()
    if args.config:
        if args.spec is not None or args.output is not None:
            parser.error("do not mix --config with low-level input/runtime flags")
        from .benchmark.core.config import configured_stage

        try:
            configured_stage(
                args.config,
                "prepare",
                scenes=args.scene,
                resume=args.resume,
                dry_run=args.dry_run,
                check_only=args.check_only,
            )
        except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as error:
            parser.exit(2, f"{error}\nSee benchmark/README.md#data-preparation.\n")
        return

    if args.dry_run or args.check_only or args.resume or args.scene:
        parser.error("these pipeline flags require --config")
    if not all([args.spec, args.output]):
        parser.error("provide --config, or --spec --output")
    plan = prepare(args.spec, args.output)
    print(f"Prepared {len(plan['scenes'])} scenes: {plan['plan_id']}")


if __name__ == "__main__":
    main()
