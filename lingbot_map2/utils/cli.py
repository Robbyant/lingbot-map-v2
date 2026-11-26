# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Run LingBot-Map2 pure causal streaming inference on an ordered RGB sequence."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import CheckpointConfig, checkpoint_paths
from .load_fn import image_paths, uniform_keyframe_ids
from ..models import load_model


def write_manifest(path, metadata):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="published pure causal model-only .pt"
    )
    parser.add_argument("--config", help="matching JSON; defaults to <checkpoint>.json")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", help="PNG/JPEG folder, natural filename order")
    inputs.add_argument("--image-list", help="ordered JSON list of image paths")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        help="persistent-frame interval after the prefix; default: ceil(N/320)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="committed temporal capacity; defaults to checkpoint configuration",
    )
    parser.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    paths = image_paths(args.images, args.image_list)
    checkpoint, config = checkpoint_paths(args.checkpoint, args.config)
    spec = CheckpointConfig.read(config)
    try:
        keyframes = uniform_keyframe_ids(
            len(paths), prefix=spec.scale_frames, interval=args.keyframe_interval
        )
        persistent = set(keyframes)
        capacity = spec.max_frames if args.max_frames is None else args.max_frames
        required = len(keyframes) + int(len(paths) - 1 not in persistent)
        if capacity < required:
            raise ValueError(
                f"--max-frames must be at least {required} for this keyframe plan"
            )
    except ValueError as error:
        parser.error(str(error))
    torch.manual_seed(42)
    model = load_model(
        checkpoint,
        config=config,
        device=args.device,
        depth=args.depth,
        width=args.width,
        height=args.height,
        compile=args.compile,
        max_frames=args.max_frames,
    )
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        model.metadata,
        status="running",
        input_count=len(paths),
        image_paths=[str(p) for p in paths],
        persistent_raw_ids=keyframes,
        keyframe_policy="Uniform320"
        if args.keyframe_interval is None
        else "fixed_interval",
        keyframe_interval=args.keyframe_interval or (len(paths) + 319) // 320,
        input_wh=[model.width, model.height],
        pose_convention="C2W",
        scene_scale="arbitrary",
        completed_outputs=[],
    )
    manifest = output / "manifest.json"
    write_manifest(manifest, metadata)
    try:
        if args.compile and len(paths) > model.config.scale_frames:
            model.warmup(paths[: min(model.config.scale_frames + 2, capacity)])
            metadata["warmup"] = model.metadata["last_warmup"]
        with model.create_stream() as stream:
            written = 0
            for raw_id, path in enumerate(paths):
                prediction = stream.online(path, persistent=raw_id in persistent)
                if prediction is None:
                    continue
                count = len(prediction["poses_c2w"])
                expected = spec.scale_frames if written == 0 else 1
                if count != expected or written + count != raw_id + 1:
                    raise RuntimeError("prediction coverage differs from input order")
                prediction["raw_indices"] = np.arange(written, written + count)
                relative = f"{written:06d}_{written + count:06d}.npz"
                np.savez(output / relative, **prediction)
                written += count
                metadata["completed_outputs"].append(relative)
                # Avoid rewriting the full input/output lists on every arrival.
                write_manifest(
                    output / "progress.json",
                    {"written_frames": written, "last_output": relative},
                )
                print(f"Saved {relative}", flush=True)
            if written != len(paths):
                raise RuntimeError("incomplete output coverage")
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_manifest(manifest, metadata)
        raise
    metadata["status"] = "complete"
    write_manifest(manifest, metadata)


if __name__ == "__main__":
    main()
