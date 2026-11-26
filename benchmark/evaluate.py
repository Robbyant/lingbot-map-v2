# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Score immutable predictions and report complete dataset aggregates only."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "benchmark"

import argparse
from importlib.metadata import version

import numpy as np

from .benchmark.core.storage import (
    read_json, write_json, sha256, identity, source_identity,
)
from .prepare import load_plan
from .benchmark.evaluation.pose import evaluate_pose, auc_metrics, THRESHOLDS


def scorer_versions(protocol):
    names = ["numpy", "scipy"]
    if protocol["kind"] == "dense":
        names.append("open3d")
    return {name: version(name) for name in names}


def verify_predictions(plan, row, folder):
    folder = Path(folder)
    receipt = read_json(folder / "COMPLETE.json")
    if (
        not receipt["complete"]
        or receipt["plan_id"] != plan["plan_id"]
        or receipt["scene"] != row["name"]
    ):
        raise ValueError("prediction belongs to a different plan/scene")
    if receipt["raw_ids"] != row["raw_ids"]:
        raise ValueError("prediction frame coverage differs from plan")
    arrays = {}
    depth = receipt["model"]["depth_enabled"]
    if type(depth) is not bool or (plan["protocol"]["kind"] == "dense" and not depth):
        raise ValueError("invalid depth output contract for this protocol")
    required = {"raw_ids.npy", "poses_c2w.npy", "intrinsics.npy", "pose_enc.npy"}
    if depth:
        required.update({"depth.npy", "depth_conf.npy"})
    if required != set(receipt["files"]):
        raise ValueError("completion receipt differs from its output contract")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or not name.endswith(".npy"):
            raise ValueError("invalid prediction file entry")
        if sha256(folder / name) != expected:
            raise ValueError(f"prediction hash mismatch: {name}")
        arrays[name[:-4]] = np.load(folder / name, allow_pickle=False, mmap_mode="r")
    np.testing.assert_array_equal(arrays["raw_ids"], row["raw_ids"])
    for key, value in arrays.items():
        if len(value) != len(row["raw_ids"]):
            raise ValueError(f"incomplete output: {key}")
        # Framewise checks keep dense-array validation memory bounded.
        if any(not np.isfinite(frame).all() for frame in value):
            raise ValueError(f"nonfinite output: {key}")
    if (
        receipt["protocol"] != plan["protocol"]["name"]
        or receipt["input_wh"] != plan["protocol"]["input_wh"]
    ):
        raise ValueError("Prediction protocol/geometry differs from plan")
    expected = [i for i, keep in zip(row["raw_ids"], row["persistent"]) if keep]
    if receipt["persistent_raw_ids"] != expected:
        raise ValueError("Prediction keyframe admission differs from plan")
    return receipt, arrays


def score_scene(plan_path, prediction_root, scene):
    plan_path = Path(plan_path)
    plan = load_plan(plan_path)
    row = next(r for r in plan["scenes"] if r["name"] == scene)
    folder = Path(prediction_root) / scene
    receipt, arrays = verify_predictions(plan, row, folder)
    protocol = plan["protocol"]
    if protocol["kind"] == "dense":
        from .benchmark.evaluation.points import validate_reference, evaluate_dense

        if sha256(row["reference"]) != row["reference_sha256"]:
            raise ValueError("shared reference changed")
        validate_reference(row["reference"], row["raw_ids"], protocol)
        metrics = evaluate_dense(arrays, row["reference"], protocol)
    else:
        gt = np.load(plan_path.parent / row["gt_file"], allow_pickle=False)
        metrics = evaluate_pose(gt, arrays["poses_c2w"])
    return dict(
        scene=scene,
        plan_id=plan["plan_id"],
        protocol=protocol["name"],
        prediction_receipt_sha256=sha256(folder / "COMPLETE.json"),
        inference_source_sha256=receipt["source_sha256"],
        scorer_source_sha256=source_identity(),
        scorer_versions=scorer_versions(protocol),
        model=receipt["model"],
        metrics=metrics,
    )


def aggregate(plan, scores):
    names = [s["scene"] for s in scores]
    if len(names) != len(set(names)):
        raise ValueError("duplicate scene scores")
    expected = set(plan["protocol"]["expected_scenes"])
    if set(names) - expected:
        raise ValueError("unexpected scene scores")
    for score in scores:
        if (
            score["plan_id"] != plan["plan_id"]
            or score["protocol"] != plan["protocol"]["name"]
        ):
            raise ValueError("mixed plans or protocols")
    signatures = {
        identity(
            {
                k: s[k]
                for k in (
                    "model",
                    "inference_source_sha256",
                    "scorer_source_sha256",
                    "scorer_versions",
                )
            }
        )
        for s in scores
    }
    if len(signatures) > 1:
        raise ValueError(
            "mixed checkpoints, precision, source or execution configurations"
        )
    missing = sorted(expected - set(names))
    result = dict(
        protocol=plan["protocol"]["name"],
        complete=not missing and not plan.get("diagnostic", False),
        diagnostic=plan.get("diagnostic", False),
        evaluated_scenes=len(names),
        expected_scenes=len(expected),
        missing_scenes=missing,
        input_scope=plan["protocol"]["input_scope"],
        input_wh=plan["protocol"]["input_wh"],
        persistence=plan["protocol"]["persistence"],
        output_scope=(
            "pose_and_geometry" if scores[0]["model"]["depth_enabled"] else "pose_only"
        )
        if scores
        else None,
        aggregation="sequence mean ATE/RPE; camera-pair-weighted micro AUC"
        if plan["protocol"]["kind"] != "dense"
        else "sequence mean CD/F1",
        input_frames=sum(
            len(row["raw_ids"]) for row in plan["scenes"] if row["name"] in names
        ),
        persistent_frames=sum(
            sum(row["persistent"]) for row in plan["scenes"] if row["name"] in names
        ),
    )
    if missing or result["diagnostic"]:
        return result  # Partial means must not be mistaken for paper means.
    metrics = [s["metrics"] for s in scores]
    if plan["protocol"]["kind"] == "dense":
        result["sequence_mean"] = {
            key: float(np.mean([m[key] for m in metrics])) for key in ("cd_m", "f1_pct")
        }
    else:
        result["sequence_mean"] = {
            key: float(np.mean([m[key] for m in metrics]))
            for key in ("ate_m", "rpe_translation_m", "rpe_rotation_deg")
        }
        statistics = dict(
            num_pairs=sum(m["auc_statistics"]["num_pairs"] for m in metrics),
            histograms={
                str(t): np.sum(
                    [m["auc_statistics"]["histograms"][str(t)] for m in metrics], axis=0
                ).tolist()
                for t in THRESHOLDS
            },
        )
        result["micro_auc"] = auc_metrics(statistics)
        result["num_pairs"] = statistics["num_pairs"]
    return result


def evaluate_run(plan_path, prediction_root, output, *, resume=False):
    plan = load_plan(plan_path)
    prediction_root, output = Path(prediction_root), Path(output)
    if not prediction_root.is_dir():
        raise FileNotFoundError(f"No predictions: {prediction_root}")
    output.mkdir(parents=True, exist_ok=resume)
    scores = []
    code_id = source_identity()
    for row in plan["scenes"]:
        folder = prediction_root / row["name"]
        if not (folder / "COMPLETE.json").exists():
            continue
        destination = output / f"{row['name']}.json"
        if destination.exists():
            if not resume:
                raise FileExistsError(destination)
            receipt, _ = verify_predictions(plan, row, folder)
            score = read_json(destination)
            if (
                score["plan_id"] != plan["plan_id"]
                or score["scene"] != row["name"]
                or score["prediction_receipt_sha256"]
                != sha256(folder / "COMPLETE.json")
                or score["scorer_source_sha256"] != code_id
                or score.get("scorer_versions") != scorer_versions(plan["protocol"])
                or score["inference_source_sha256"] != receipt["source_sha256"]
            ):
                raise ValueError(f"Saved score changed provenance: {destination}")
            if (
                plan["protocol"]["kind"] == "dense"
                and sha256(row["reference"]) != row["reference_sha256"]
            ):
                raise ValueError("Shared reference changed")
        else:
            score = score_scene(plan_path, prediction_root, row["name"])
            write_json(destination, score)
        scores.append(score)
    report = aggregate(plan, scores)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--predictions")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", help="base YAML under benchmark/configs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved settings without data access or CUDA",
    )
    parser.add_argument("--scene", action="append")
    args = parser.parse_args()
    if args.config:
        if (
            args.plan is not None
            or args.predictions is not None
            or args.output is not None
        ):
            parser.error("do not mix --config with low-level input/runtime flags")
        from .benchmark.core.config import configured_stage

        try:
            configured_stage(
                args.config,
                "evaluate",
                scenes=args.scene,
                resume=args.resume,
                dry_run=args.dry_run,
            )
        except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as error:
            parser.exit(2, f"{error}\nSee benchmark/README.md#data-preparation.\n")
        return

    if args.dry_run or args.scene:
        parser.error("these pipeline flags require --config")
    if not all([args.plan, args.predictions, args.output]):
        parser.error("provide --config, or --plan --predictions --output")
    print(evaluate_run(args.plan, args.predictions, args.output, resume=args.resume))


if __name__ == "__main__":
    main()
