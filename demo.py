# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Run Online inference or load saved predictions, then launch the Viser viewer."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from lingbot_map2.utils.load_fn import (
    cover_resize,
    exact_keyframe_ids,
    image_paths,
    uniform_keyframe_ids,
)

DEFAULT_DEMO_KEYFRAMES = 320


def select_demo_keyframes(
    count,
    *,
    prefix,
    num_keyframes=None,
    keyframe_interval=None,
    default_num_keyframes=DEFAULT_DEMO_KEYFRAMES,
):
    """Select persistent KV-cache keyframes for demo.py Online inference."""
    if num_keyframes is not None and keyframe_interval is not None:
        raise ValueError("cannot set both --num-keyframes and --keyframe-interval")
    if keyframe_interval is not None:
        return uniform_keyframe_ids(count, prefix=prefix, interval=keyframe_interval)
    target = default_num_keyframes if num_keyframes is None else num_keyframes
    return exact_keyframe_ids(count, prefix=prefix, num_keyframes=target)


def run_online_inference(model, paths, keyframes):
    """Run pure causal Online streaming inference and collect full predictions."""
    pending_rgbs = []
    images, poses, intrinsics, depths, confs = [], [], [], [], []
    with model.create_stream() as stream:
        for raw_id, path in enumerate(paths):
            tensor = model.prepare(path)[0, 0]
            rgb = (
                np.rint(tensor.permute(1, 2, 0).numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            pending_rgbs.append(rgb)
            prediction = stream.online(tensor, persistent=raw_id in keyframes)
            if prediction is None:
                if len(pending_rgbs) >= model.config.scale_frames:
                    raise RuntimeError("missing initialization predictions")
                continue
            if len(prediction["poses_c2w"]) != len(pending_rgbs):
                raise RuntimeError("predictions do not cover the supplied RGB frames")
            images.extend(pending_rgbs)
            poses.append(prediction["poses_c2w"])
            intrinsics.append(prediction["intrinsics"])
            depths.append(prediction["depth"])
            confs.append(prediction["depth_conf"])
            pending_rgbs.clear()
    if len(images) != len(paths):
        raise RuntimeError("incomplete Online prediction coverage")
    return {
        "images": np.stack(images, axis=0),
        "poses_c2w": np.concatenate(poses, axis=0),
        "intrinsics": np.concatenate(intrinsics, axis=0),
        "depth": np.concatenate(depths, axis=0),
        "depth_conf": np.concatenate(confs, axis=0),
    }


def load_saved_predictions(folder):
    """Load complete infer.py predictions and their aligned processed RGB frames."""
    from PIL import Image

    folder = Path(folder)
    metadata = json.loads((folder / "manifest.json").read_text())
    if metadata.get("status") != "complete" or not metadata.get("depth_enabled"):
        raise ValueError("viewer requires a complete infer.py run with depth enabled")
    paths, geometry = metadata["image_paths"], metadata["input_wh"]
    images, poses, intrinsics, depths, confs = [], [], [], [], []
    expected_id = 0
    for name in metadata["completed_outputs"]:
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("invalid prediction file in manifest")
        with np.load(folder / name, allow_pickle=False) as prediction:
            raw_indices = prediction["raw_indices"]
            for raw_id in raw_indices:
                if raw_id != expected_id or raw_id >= len(paths):
                    raise ValueError(
                        "saved predictions do not cover RGB inputs in order"
                    )
                with Image.open(paths[raw_id]) as source:
                    rgb = cover_resize(
                        np.asarray(source.convert("RGB"), dtype=np.float32), geometry
                    )
                images.append(np.rint(rgb).clip(0, 255).astype(np.uint8))
                expected_id += 1
            poses.append(np.array(prediction["poses_c2w"]))
            intrinsics.append(np.array(prediction["intrinsics"]))
            depths.append(np.array(prediction["depth"]))
            confs.append(np.array(prediction["depth_conf"]))
    if expected_id != len(paths):
        raise ValueError("saved prediction coverage is incomplete")
    return (
        {
            "images": np.stack(images, axis=0),
            "poses_c2w": np.concatenate(poses, axis=0),
            "intrinsics": np.concatenate(intrinsics, axis=0),
            "depth": np.concatenate(depths, axis=0),
            "depth_conf": np.concatenate(confs, axis=0),
        },
        paths,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint", "--model_path", help="pure causal model-only .pt"
    )
    source.add_argument("--predictions", help="complete output folder from infer.py")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--images", "--image_folder", help="naturally ordered RGB folder"
    )
    inputs.add_argument("--image-list", help="ordered JSON list of RGB paths")
    parser.add_argument("--config", help="checkpoint JSON; default: <checkpoint>.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    keyframe_policy = parser.add_mutually_exclusive_group()
    keyframe_policy.add_argument(
        "--num-keyframes",
        "--num_keyframes",
        type=int,
        default=None,
        help=(
            "target persistent keyframes in the KV cache "
            f"(default: {DEFAULT_DEMO_KEYFRAMES} when --keyframe-interval is unset; "
            "must be >= scale_frames, which retains only the initialization prefix; "
            "non-keyframes are predicted without entering the KV cache)"
        ),
    )
    keyframe_policy.add_argument(
        "--keyframe-interval",
        "--keyframe_interval",
        type=int,
        default=None,
        help="fixed persistent-frame interval after the prefix (overrides --num-keyframes)",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--depth-stride",
        "--depth_stride",
        type=int,
        default=1,
        help="project depth to points every N frames (1 = every frame)",
    )
    parser.add_argument("--point-size", "--point_size", type=float, default=0.00001)
    parser.add_argument("--camera-size", "--camera_size", type=float, default=0.01)
    parser.add_argument(
        "--conf-threshold", "--conf_threshold", type=float, default=1.5
    )
    parser.add_argument(
        "--downsample-factor", "--downsample_factor", type=int, default=10
    )
    parser.add_argument("--mask-sky", "--mask_sky", action="store_true")
    parser.add_argument(
        "--sky-model",
        "--sky_model",
        help="skyseg.onnx; downloaded to ~/.cache/lingbot-map2/skyseg.onnx if omitted",
    )
    parser.add_argument(
        "--sky-mask-dir",
        "--sky_mask_dir",
        help="optional directory for cached sky masks",
    )
    parser.add_argument(
        "--sky-mask-visualization-dir",
        "--sky_mask_visualization_dir",
        help="optional directory to save original | mask | overlay previews",
    )
    args = parser.parse_args(argv)
    if args.depth_stride < 1:
        parser.error("--depth-stride must be positive")
    if args.num_keyframes is not None and args.num_keyframes < 1:
        parser.error("--num-keyframes must be positive")
    if args.keyframe_interval is not None and args.keyframe_interval < 1:
        parser.error("--keyframe-interval must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.checkpoint and not (args.images or args.image_list):
        parser.error("--checkpoint requires --images or --image-list")
    if args.predictions and any(
        v is not None
        for v in (
            args.images,
            args.image_list,
            args.config,
            args.width,
            args.height,
            args.num_keyframes,
            args.keyframe_interval,
            args.max_frames,
        )
    ):
        parser.error("--predictions uses the saved run's images and configuration")
    try:
        from lingbot_map2.vis import PointCloudViewer
    except ModuleNotFoundError as error:
        if error.name in {"viser", "matplotlib", "onnxruntime", "cv2", "trimesh"}:
            parser.error(
                'install visualization dependencies with: pip install -e ".[vis]"'
            )
        raise

    if args.checkpoint:
        from lingbot_map2 import load_model
        from lingbot_map2.utils.checkpoint import CheckpointConfig, checkpoint_paths

        paths = image_paths(args.images, args.image_list)
        checkpoint, config = checkpoint_paths(args.checkpoint, args.config)
        spec = CheckpointConfig.read(config)
        try:
            keyframes = set(
                select_demo_keyframes(
                    len(paths),
                    prefix=spec.scale_frames,
                    num_keyframes=args.num_keyframes,
                    keyframe_interval=args.keyframe_interval,
                )
            )
        except ValueError as error:
            parser.error(str(error))
        capacity = spec.max_frames if args.max_frames is None else args.max_frames
        required = len(keyframes) + int(len(paths) - 1 not in keyframes)
        if capacity < required:
            parser.error(
                f"--max-frames must be at least {required} for these keyframes"
            )
        print(
            f"Running Online inference on {len(paths)} frames "
            f"({len(keyframes)} persistent keyframes in KV cache, "
            f"{len(paths) - len(keyframes)} non-keyframes).",
            flush=True,
        )
        model = load_model(
            checkpoint,
            config=config,
            device=args.device,
            width=args.width,
            height=args.height,
            compile=args.compile,
            max_frames=args.max_frames,
            depth=True,
        )
        if args.compile and len(paths) > spec.scale_frames:
            model.warmup(paths[: min(spec.scale_frames + 2, capacity)])
        pred_dict = run_online_inference(model, paths, keyframes)
        str_paths = [str(p) for p in paths]
        device = model.device
        del model
        gc.collect()
        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()
    else:
        pred_dict, str_paths = load_saved_predictions(args.predictions)

    viewer = PointCloudViewer(
        pred_dict=pred_dict,
        host=args.host,
        port=args.port,
        point_size=args.point_size,
        camera_size=args.camera_size,
        vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        depth_stride=args.depth_stride,
        mask_sky=args.mask_sky,
        image_folder=args.images,
        image_paths=str_paths,
        skyseg_model_path=args.sky_model,
        sky_mask_dir=args.sky_mask_dir,
        sky_mask_visualization_dir=args.sky_mask_visualization_dir,
    )
    print(
        f"Open http://{args.host}:{viewer.server.get_port()} in your browser.",
        flush=True,
    )
    print("Finished. The viewer stays open; press Ctrl+C to exit.", flush=True)
    try:
        viewer.run()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
