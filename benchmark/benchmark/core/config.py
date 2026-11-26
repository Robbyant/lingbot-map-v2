# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Shared configuration and stage dispatch for the three benchmark entrypoints."""

import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from .storage import read_json, sha256, write_json
from ...datasets.adapters import inspect_dataset, prepare_native
from ...prepare import load_plan, prepare
from ...datasets.protocols import PROTOCOLS, get_protocol


def read_configs(path):
    """Resolve Map1-style base/dataset/method YAMLs, restricted to this release."""
    import yaml

    def read(file):
        value = yaml.safe_load(file.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"Expected a mapping in {file}")
        return value

    def resolve(file, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Expected a nonempty path in {file}")
        return str((file.parent / Path(value).expanduser()).resolve())

    path = Path(path).resolve(strict=True)
    base = read(path)
    if set(base) - {"workspace", "datasets", "methods", "config_root"}:
        raise ValueError(
            "Base config supports workspace, datasets, methods and optional config_root"
        )
    if base.get("methods") != ["lingbot_map2"]:
        raise ValueError("This release supports methods: [lingbot_map2] only")
    names = base.get("datasets")
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(n, str) or not n or Path(n).name != n or n in (".", "..")
            for n in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "datasets must be a nonempty list of unique configuration names"
        )
    config_root = (
        Path(resolve(path, base["config_root"]))
        if "config_root" in base
        else path.parent
    )
    method_path = config_root / "methods/lingbot_map2.yaml"
    method = read(method_path)
    allowed_method = {
        "model",
        "_mode",
        "_checkpoint",
        "_config",
        "_device",
        "_compile",
        "_depth",
    }
    if (
        set(method) - allowed_method
        or method.get("model") != "GCRStream"
        or method.get("_mode") != "online"
    ):
        raise ValueError("Method config must select GCRStream in online mode")
    compile_model = method.get("_compile", True)
    if type(compile_model) is not bool:
        raise ValueError("_compile must be a YAML Boolean")
    depth = method.get("_depth", True)
    if type(depth) is not bool:
        raise ValueError("_depth must be a YAML Boolean")
    checkpoint = (
        resolve(method_path, method["_checkpoint"])
        if method.get("_checkpoint")
        else None
    )
    sidecar = resolve(method_path, method["_config"]) if method.get("_config") else None
    workspace = Path(resolve(path, base["workspace"]))
    configs = []
    for name in names:
        file = config_root / "datasets" / f"{name}.yaml"
        data = read(file)
        if set(data) - {
            "protocol",
            "raw_data_root",
            "scenes",
            "reference_root",
            "camera_index",
            "input_scope",
        }:
            raise ValueError(f"Unknown dataset configuration fields: {file}")
        if data.get("protocol") not in PROTOCOLS:
            raise ValueError(f"Choose a supported paper protocol in {file}")
        protocol = get_protocol(data["protocol"], data.get("input_scope"))
        if protocol["kind"] == "dense" and not depth:
            raise ValueError("Dense reconstruction requires _depth: true")
        scenes = data.get("scenes")
        if scenes is not None and (
            not isinstance(scenes, list)
            or not scenes
            or any(not isinstance(n, str) for n in scenes)
        ):
            raise ValueError("scenes must be null or a nonempty list of scene names")
        camera_index = data.get("camera_index", "camera/frames.jsonl")
        if (
            not isinstance(camera_index, str)
            or not camera_index
            or Path(camera_index).is_absolute()
            or ".." in Path(camera_index).parts
        ):
            raise ValueError("camera_index must be relative to each scene")
        configs.append(
            dict(
                protocol=data["protocol"],
                raw_data_root=resolve(file, data["raw_data_root"]),
                workspace=str(workspace / name),
                scenes=scenes,
                reference_root=resolve(file, data["reference_root"])
                if data.get("reference_root")
                else None,
                camera_index=camera_index,
                checkpoint=checkpoint,
                checkpoint_config=sidecar,
                device=method.get("_device", "cuda:0"),
                compile=compile_model,
                depth=depth,
                input_scope=protocol["input_scope"],
            )
        )
    return configs


def data_settings(config):
    return {
        k: config[k]
        for k in (
            "protocol",
            "raw_data_root",
            "scenes",
            "reference_root",
            "camera_index",
            "input_scope",
        )
    }


def data_arguments(config):
    return dict(
        scenes=config["scenes"],
        reference_root=config["reference_root"],
        camera_index=config["camera_index"],
    )


def prepare_stage(config, *, resume=False):
    root = Path(config["workspace"])
    receipt = root / "PREPARED.json"
    plan_path = root / "plan/plan.json"
    if receipt.exists():
        if not resume:
            raise FileExistsError(
                f"Prepared workspace exists: {root}. Use --resume to verify and reuse it."
            )
        saved = read_json(receipt)
        if saved["data_settings"] != data_settings(config):
            raise ValueError("Data configuration changed; use a new workspace")
        plan = load_plan(plan_path)
        if (
            saved["plan_sha256"] != sha256(plan_path)
            or saved["plan_id"] != plan["plan_id"]
        ):
            raise ValueError("Prepared plan changed")
        # Recheck GT associations, reference revision and original file paths.
        rows = inspect_dataset(
            config["protocol"], config["raw_data_root"], **data_arguments(config)
        )
        import numpy as np
        from ...methods.lingbot_map2 import frames

        for (raw, poses), frozen in zip(rows, plan["scenes"], strict=True):
            if raw["name"] != frozen["name"] or raw["raw_count"] != frozen["raw_count"]:
                raise ValueError("Scene order/coverage changed")
            gt = np.load(plan_path.parent / frozen["gt_file"], allow_pickle=False)
            np.testing.assert_array_equal(poses[frozen["raw_ids"]], gt)
            if "images" in raw:
                if [raw["images"][i] for i in frozen["raw_ids"]] != frozen["images"]:
                    raise ValueError("RGB paths changed")
                for _ in frames(frozen):
                    pass  # File hashes only; images are not decoded by this generator.
            elif (
                raw["video"] != frozen["video"]
                or sha256(raw["video"]) != frozen["video_sha256"]
            ):
                raise ValueError("Video source changed")
            if (
                "reference" in raw
                and sha256(raw["reference"]) != frozen["reference_sha256"]
            ):
                raise ValueError("Dense reference changed")
        print(f"Reusing verified plan: {plan_path}", flush=True)
        return
    if (root / "data").exists() or (root / "plan").exists():
        raise FileExistsError(
            "Incomplete preparation is preserved; choose a new workspace"
        )
    prepare_native(
        config["protocol"],
        config["raw_data_root"],
        root / "data",
        input_scope=config["input_scope"],
        **data_arguments(config),
    )
    plan = prepare(root / "data/dataset.json", root / "plan")
    write_json(
        receipt,
        dict(
            data_settings=data_settings(config),
            plan_id=plan["plan_id"],
            plan_sha256=sha256(plan_path),
        ),
    )
    print(f"Prepared {len(plan['scenes'])} scenes: {plan_path}", flush=True)


def prepared_plan(config):
    root = Path(config["workspace"])
    saved = read_json(root / "PREPARED.json")
    if saved["data_settings"] != data_settings(config):
        raise ValueError(
            "Configuration differs from preparation; keep the same config or use a new workspace"
        )
    plan_path = root / "plan/plan.json"
    plan = load_plan(plan_path)
    if saved["plan_id"] != plan["plan_id"] or saved["plan_sha256"] != sha256(plan_path):
        raise ValueError("Prepared plan differs from its receipt")
    return plan_path, plan


def run_stage(config, *, resume=False, retry_failed=False):
    if retry_failed and not resume:
        raise ValueError("--retry-failed requires --resume")
    plan_path, plan = prepared_plan(config)
    checkpoint = config["checkpoint"]
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            "Set checkpoint to the released pure causal weights; data preparation does not require weights"
        )
    from lingbot_map2.utils.checkpoint import resolve_checkpoint
    from .storage import source_identity
    from ...evaluate import verify_predictions

    _, spec, _ = resolve_checkpoint(checkpoint, config["checkpoint_config"])
    expected_compile = (
        "reduce-overhead; lazy shape-dependent capture" if config["compile"] else False
    )
    code_id = source_identity()
    from importlib.metadata import version
    import torch

    versions = {name: version(name) for name in ("torch", "numpy", "flashinfer-python")}
    capacity = max(sum(row["persistent"]) + 1 for row in plan["scenes"])
    pending, failed = [], []
    predictions = Path(config["workspace"]) / "predictions"
    for row in plan["scenes"]:
        folder = predictions / row["name"]
        if not folder.exists():
            pending.append(row["name"])
            continue
        if not resume:
            raise FileExistsError(
                f"Existing prediction: {folder}; use --resume to verify completed scenes"
            )
        complete = (folder / "COMPLETE.json").exists()
        if complete:
            receipt, _ = verify_predictions(plan, row, folder)
        else:
            if not retry_failed:
                raise ValueError(
                    f"{row['name']}: incomplete prediction; use --resume --retry-failed "
                    "to archive this attempt and retry the scene"
                )
            if not (folder / "ATTEMPT.json").is_file():
                raise ValueError(
                    f"{row['name']}: missing attempt identity; use a new workspace"
                )
            receipt = read_json(folder / "ATTEMPT.json")
            if (
                receipt["plan_id"] != plan["plan_id"]
                or receipt["scene"] != row["name"]
                or receipt["raw_ids"] != row["raw_ids"]
                or receipt["input_wh"] != plan["protocol"]["input_wh"]
                or receipt["protocol"] != plan["protocol"]["name"]
            ):
                raise ValueError(f"{row['name']}: attempt belongs to another plan")
        metadata = receipt["model"]
        if (
            receipt["source_sha256"] != code_id
            or metadata["checkpoint_sha256"] != spec.sha256
            or metadata["checkpoint_config"] != spec.to_dict()
            or metadata["compile"] != expected_compile
            or metadata["depth_enabled"] != config["depth"]
            or metadata["temporal_capacity"] != capacity
            or metadata.get("versions") != versions
            or metadata.get("matmul_tf32") is not False
            or metadata.get("cudnn_tf32") != torch.backends.cudnn.allow_tf32
        ):
            raise ValueError(
                f"{row['name']}: checkpoint/runtime/source differs; choose a new workspace"
            )
        if complete:
            print(f"Skipping verified completed scene: {row['name']}", flush=True)
        else:
            failed.append(folder)
            pending.append(row["name"])
    if not pending:
        print("All scenes already complete; no CUDA model loaded.", flush=True)
        return
    # Validate all completed and failed identities before moving any attempt.
    # Preserve old predictions/scores and never infer a resumable KV state.
    for folder in failed:
        archive = (
            Path(config["workspace"]) / "failed_attempts" / folder.name / uuid4().hex
        )
        archive.mkdir(parents=True)
        folder.rename(archive / "prediction")
        score = Path(config["workspace"]) / "scores" / f"{folder.name}.json"
        if score.exists():
            score.rename(archive / "score.json")
        report = Path(config["workspace"]) / "scores/report.json"
        if report.exists():
            report.rename(archive / "previous_report.json")
        print(f"Archived incomplete attempt: {archive}", flush=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "run.py"),
        "--plan",
        str(plan_path),
        "--checkpoint",
        checkpoint,
        "--output",
        str(predictions),
        "--device",
        config["device"],
        "--compile" if config["compile"] else "--no-compile",
        "--depth" if config["depth"] else "--no-depth",
    ]
    if config["checkpoint_config"]:
        command += ["--checkpoint-config", config["checkpoint_config"]]
    for scene in pending:
        command += ["--scene", scene]
    subprocess.run(command, check=True)


def evaluate_stage(config, *, resume=False):
    from ...evaluate import evaluate_run

    plan_path, _ = prepared_plan(config)
    root = Path(config["workspace"])
    return evaluate_run(plan_path, root / "predictions", root / "scores", resume=resume)


def configured_stage(
    path,
    stage,
    *,
    scenes=None,
    resume=False,
    dry_run=False,
    check_only=False,
    retry_failed=False,
):
    if retry_failed and (stage != "run" or not resume):
        raise ValueError("--retry-failed is available only for run with --resume")
    configs = read_configs(path)
    for config in configs:
        if scenes is not None:
            config["scenes"] = scenes
        protocol = get_protocol(config["protocol"], config["input_scope"])
        selected = config["scenes"] or protocol["expected_scenes"]
        if len(selected) != len(set(selected)) or set(selected) - set(
            protocol["expected_scenes"]
        ):
            raise ValueError(
                "Unknown/duplicate scene; see benchmark/configs/scenes.json"
            )
        if dry_run:
            print(
                json.dumps(
                    dict(
                        stage=stage,
                        resume=resume,
                        retry_failed=retry_failed,
                        input_wh=protocol["input_wh"],
                        selected_scenes=selected,
                        **config,
                    ),
                    indent=2,
                )
            )
            continue
        if check_only:
            rows = inspect_dataset(
                config["protocol"], config["raw_data_root"], **data_arguments(config)
            )
            print(
                f"{config['protocol']}: data ready, {len(rows)}/{len(protocol['expected_scenes'])} scenes, {sum(r['raw_count'] for r, _ in rows)} raw frames."
            )
            continue
        # One pipeline writer per workspace; check/dry-run never create a lock.
        import fcntl

        root = Path(config["workspace"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".pipeline.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another pipeline stage owns {root}") from error
            if stage == "prepare":
                prepare_stage(config, resume=resume)
            elif stage == "run":
                run_stage(config, resume=resume, retry_failed=retry_failed)
            elif stage == "evaluate":
                print(evaluate_stage(config, resume=resume))
            else:
                raise ValueError(f"Unknown pipeline stage: {stage}")
