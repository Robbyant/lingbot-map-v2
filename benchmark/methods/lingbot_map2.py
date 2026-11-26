# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Pure Online inference adapter for the native LingBot-Map 2.0 model."""

from pathlib import Path
import importlib.metadata
import time

import numpy as np

from ..benchmark.core.storage import sha256, write_json


def frames(row):
    """Decode in chronological order; video indices are never approximate seeks."""
    if "images" in row:
        for path, expected in zip(row["images"], row["image_sha256"]):
            if sha256(path) != expected:
                raise ValueError(f"RGB changed after planning: {path}")
            yield path
        return
    import cv2

    if sha256(row["video"]) != row["video_sha256"]:
        raise ValueError("video changed after planning")
    capture = cv2.VideoCapture(row["video"])
    wanted = iter(row["raw_ids"])
    next_id = next(wanted)
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if index == next_id:
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                next_id = next(wanted, None)
            index += 1
        if index != row["raw_count"] or next_id is not None:
            raise ValueError(
                f"decoded video length {index} differs from frozen raw_count"
            )
    finally:
        capture.release()


def run_scene(model, row, protocol, output, plan_id, code_id):
    output = Path(output)
    if (model.width, model.height) != tuple(protocol["input_wh"]):
        raise ValueError("model geometry does not match the plan")
    if (model.config.scale_frames, model.config.sliding_window) != (8, 64):
        raise ValueError("paper presets require prefix8 and window64")
    depth = model._native.depth_head is not None
    if protocol["kind"] == "dense" and not depth:
        raise ValueError("dense reconstruction requires geometry outputs")
    required = {"poses_c2w", "intrinsics", "pose_enc"}
    if depth:
        required.update({"depth", "depth_conf"})
    attempt = dict(
        scene=row["name"],
        plan_id=plan_id,
        protocol=protocol["name"],
        input_wh=protocol["input_wh"],
        source_sha256=code_id,
        model=model.metadata,
        raw_ids=row["raw_ids"],
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "ATTEMPT.json", attempt)
    n = len(row["raw_ids"])
    arrays, written, received = {}, 0, 0
    started = time.monotonic()
    try:
        with model.create_stream(output_device="cpu") as stream:
            for index, frame in enumerate(frames(row)):
                received += 1
                prediction = stream.online(frame, persistent=row["persistent"][index])
                if prediction is None:
                    if index >= protocol["prefix"] - 1:
                        raise RuntimeError(
                            "model omitted an output after initialization"
                        )
                    continue
                if required - prediction.keys():
                    raise RuntimeError("incomplete model outputs")
                count = len(prediction["poses_c2w"])
                expected = protocol["prefix"] if written == 0 else 1
                if count != expected or written + count > n:
                    raise RuntimeError("prediction frame count differs from input")
                for key in sorted(required):
                    value = np.asarray(prediction[key])
                    if value.shape[0] != count or not np.isfinite(value).all():
                        raise ValueError("nonfinite or misaligned prediction")
                    if key not in arrays:
                        arrays[key] = np.lib.format.open_memmap(
                            output / f"{key}.npy",
                            mode="w+",
                            dtype=value.dtype,
                            shape=(n, *value.shape[1:]),
                        )
                    arrays[key][written : written + count] = value
                written += count
        if received != n or written != n:
            raise RuntimeError("incomplete sequence; no completion receipt written")
        for value in arrays.values():
            value.flush()
        np.save(output / "raw_ids.npy", np.asarray(row["raw_ids"], dtype=np.int64))
        receipt = dict(
            complete=True,
            scene=row["name"],
            plan_id=plan_id,
            protocol=protocol["name"],
            input_wh=protocol["input_wh"],
            pose_convention="C2W",
            source_sha256=code_id,
            model=model.metadata,
            raw_ids=row["raw_ids"],
            persistent_raw_ids=[
                i for i, keep in zip(row["raw_ids"], row["persistent"]) if keep
            ],
            files={p.name: sha256(p) for p in sorted(output.glob("*.npy"))},
            wall_seconds_including_io=time.monotonic() - started,
            timing_scope="accuracy run including decoding, hashing, inference, CPU output and saving; not a warmed speed benchmark",
        )
        write_json(output / "COMPLETE.json", receipt)
        return receipt
    except BaseException as error:
        write_json(
            output / "FAILED.json",
            dict(type=type(error).__name__, error=str(error), written_frames=written),
        )
        raise


def load_model(
    plan, checkpoint, *, config=None, depth=True, device="cuda:0", compile=True
):
    """Load the causal model with the frozen plan geometry and capacity."""
    from lingbot_map2 import load_model as load_stream_model
    import torch

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False

    # Reserve one temporal position for a final transient after the last keyframe.
    capacity = max(sum(r["persistent"]) + 1 for r in plan["scenes"])
    width, height = plan["protocol"]["input_wh"]
    model = load_stream_model(
        checkpoint,
        config=config,
        depth=depth,
        device=device,
        width=width,
        height=height,
        compile=compile,
        max_frames=capacity,
    )
    model.metadata["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "flashinfer-python")
    }
    model.metadata["matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
    model.metadata["cudnn_tf32"] = torch.backends.cudnn.allow_tf32
    return model
