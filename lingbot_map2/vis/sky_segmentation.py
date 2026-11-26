# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 The Robbyant Team Authors. All rights reserved.
# Adapted from Robbyant/lingbot-map, lingbot_map/vis/sky_segmentation.py
# (b9dd1b1de535e6d36e777fb42b8433c781bfd783). See LICENSES/THIRD_PARTY.md.

"""Sky segmentation utilities for filtering sky points from point clouds."""

import glob
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple
from urllib.request import urlopen

import cv2
import numpy as np

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

_SKYSEG_INPUT_SIZE = (320, 320)
# Non-sky confidence threshold; > 0.875 keeps raw skyseg scores < 32/255, matching V1/VGGT.
_SKYSEG_SOFT_THRESHOLD = 0.875
_SKYSEG_CACHE_VERSION = "imagenet_norm_softmap_inverted_v6"
_SKYSEG_MODEL_URL = (
    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
)
_DEFAULT_SKYSEG_PATH = Path.home() / ".cache/lingbot-map2/skyseg.onnx"


def _get_cache_version_path(sky_mask_dir: str) -> str:
    return os.path.join(sky_mask_dir, ".skyseg_cache_version")


def _prepare_sky_mask_cache(sky_mask_dir: Optional[str]) -> bool:
    """Ensure the sky mask cache directory exists; return True if cached masks are stale."""
    if sky_mask_dir is None:
        return False
    os.makedirs(sky_mask_dir, exist_ok=True)
    version_path = _get_cache_version_path(sky_mask_dir)
    refresh_cache = True
    if os.path.exists(version_path):
        with open(version_path, "r", encoding="utf-8") as f:
            refresh_cache = f.read().strip() != _SKYSEG_CACHE_VERSION
    if refresh_cache:
        with open(version_path, "w", encoding="utf-8") as f:
            f.write(_SKYSEG_CACHE_VERSION)
    return refresh_cache


def download_skyseg_model(output_path: Optional[str] = None) -> str:
    """Download the sky segmentation ONNX model from Hugging Face atomically."""
    path = Path(output_path).expanduser() if output_path else _DEFAULT_SKYSEG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(_SKYSEG_MODEL_URL, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                while chunk := response.read(1 << 20):
                    output.write(chunk)
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(path)


def _ensure_skyseg_model(skyseg_model_path: Optional[str]) -> str:
    """Download the skyseg model when the requested path is not a regular file."""
    path = (
        Path(skyseg_model_path).expanduser()
        if skyseg_model_path
        else _DEFAULT_SKYSEG_PATH
    )
    if path.is_file():
        return str(path)
    try:
        download_skyseg_model(str(path))
    except Exception as error:
        raise RuntimeError(
            f"Failed to download the sky segmentation model to {path} "
            f"from {_SKYSEG_MODEL_URL}: {error}. Download it manually from "
            f"{_SKYSEG_MODEL_URL} and pass --sky-model /path/to/skyseg.onnx."
        ) from error
    if not path.is_file():
        raise RuntimeError(
            f"Sky segmentation model download to {path} did not create a file."
        )
    return str(path)


def _create_ort_session(model_path: str):
    if onnxruntime is None:
        raise ModuleNotFoundError(
            "onnxruntime is required for --mask-sky; install with: pip install -e '.[vis]'",
            name="onnxruntime",
        )
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run_skyseg(
    onnx_session,
    input_size: Tuple[int, int],
    image: np.ndarray,
) -> np.ndarray:
    """Run ONNX sky segmentation on a BGR image and return an 8-bit score map."""
    resize_image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x / 255.0 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[1], input_size[0]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    denom = max(max_value - min_value, 1e-8)
    onnx_result = (onnx_result - min_value) / denom
    onnx_result *= 255.0
    return onnx_result.astype(np.uint8)


def _mask_to_float(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        return mask.astype(np.float32) / 255.0
    mask = mask.astype(np.float32)
    if mask.size == 0:
        return mask
    return np.clip(mask, 0.0, 1.0)


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype == np.uint8:
        return mask
    mask = mask.astype(np.float32)
    if mask.size > 0 and mask.max() <= 1.0:
        mask = mask * 255.0
    return np.clip(mask, 0.0, 255.0).astype(np.uint8)


def _result_map_to_non_sky_conf(result_map: np.ndarray) -> np.ndarray:
    # The raw skyseg map is higher on sky and lower on non-sky.
    return 1.0 - _mask_to_float(result_map)


def _image_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected image with shape (H, W, 3) or (3, H, W), got {image.shape}"
        )

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size > 0 and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    return image


def segment_sky_from_array(
    image: np.ndarray,
    skyseg_session,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Segment sky from an image array and return a [0, 1] non-sky confidence map."""
    image_rgb = _image_to_rgb_uint8(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result_map = run_skyseg(skyseg_session, _SKYSEG_INPUT_SIZE, image_bgr)
    result_map = cv2.resize(
        result_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )
    return _result_map_to_non_sky_conf(result_map)


def segment_sky(
    image_path: str,
    skyseg_session,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Segment sky from an image file and return a [0, 1] non-sky confidence map."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    result_map = run_skyseg(skyseg_session, _SKYSEG_INPUT_SIZE, image)
    result_map = cv2.resize(
        result_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    mask = _result_map_to_non_sky_conf(result_map)

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if not cv2.imwrite(output_path, _mask_to_uint8(mask)):
            raise OSError(f"Failed to write sky mask: {output_path}")

    return mask


def _list_image_files(image_folder: str) -> list[str]:
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return [
        f for f in image_files if os.path.splitext(f.lower())[1] in image_extensions
    ]


def _get_model_fingerprint(model_path: str) -> str:
    """Compute a short fingerprint combining resolved model path and file bytes."""
    resolved = str(Path(model_path).resolve(strict=False))
    hasher = hashlib.sha256()
    hasher.update(b"model_path:")
    hasher.update(resolved.encode("utf-8"))
    if os.path.isfile(model_path):
        hasher.update(b"\0model_bytes:")
        with open(model_path, "rb") as f:
            while chunk := f.read(1 << 20):
                hasher.update(chunk)
    return hasher.hexdigest()[:16]


def _get_mask_filename(
    image_paths: Optional[list[str]],
    index: int,
    image_rgb: Optional[np.ndarray] = None,
    model_fingerprint: Optional[str] = None,
) -> str:
    """Build a collision-resistant cache filename incorporating model, path, and content."""
    hasher = hashlib.sha256()
    has_path = image_paths is not None and index < len(image_paths)
    if has_path:
        path = Path(image_paths[index])
        resolved = str(path.resolve(strict=False))
        hasher.update(b"path:")
        hasher.update(resolved.encode("utf-8"))
        stem = path.stem
    else:
        stem = f"frame_{index:06d}"

    if image_rgb is not None:
        contig = np.ascontiguousarray(image_rgb)
        hasher.update(f"\0shape:{contig.shape}\0".encode("utf-8"))
        hasher.update(contig.data)

    if model_fingerprint:
        hasher.update(b"\0model:")
        hasher.update(model_fingerprint.encode("utf-8"))

    if has_path or image_rgb is not None or model_fingerprint:
        return f"{stem}_{hasher.hexdigest()[:16]}.png"
    return f"{stem}.png"


def _get_visualization_filename(image_paths: Optional[list[str]], index: int) -> str:
    if image_paths is not None and index < len(image_paths):
        return f"{index:06d}_{Path(image_paths[index]).stem}.png"
    return f"frame_{index:06d}.png"


def _save_sky_mask_visualization(
    image: np.ndarray,
    sky_mask: np.ndarray,
    output_path: str,
) -> None:
    image_rgb = _image_to_rgb_uint8(image)
    if sky_mask.shape[:2] != image_rgb.shape[:2]:
        sky_mask = cv2.resize(
            sky_mask.astype(np.float32),
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_uint8 = _mask_to_uint8(sky_mask)
    mask_rgb = np.repeat(mask_uint8[..., None], 3, axis=2)
    overlay = image_rgb.astype(np.float32).copy()
    sky_pixels = (
        ~sky_mask
        if sky_mask.dtype == bool
        else _mask_to_float(sky_mask) <= _SKYSEG_SOFT_THRESHOLD
    )
    overlay[sky_pixels] = (
        overlay[sky_pixels] * 0.35 + np.array([255, 64, 64], dtype=np.float32) * 0.65
    )
    overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    panel = np.concatenate([image_rgb, mask_rgb, overlay], axis=1)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def load_or_create_sky_masks(
    image_folder: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    images: Optional[np.ndarray] = None,
    skyseg_model_path: Optional[str] = None,
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    num_frames: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Load cached sky masks or generate them with the ONNX model."""
    if onnxruntime is None:
        print("Warning: onnxruntime not available, skipping sky segmentation")
        return None

    if image_folder is None and image_paths is None and images is None:
        print(
            "Warning: Neither image_folder/image_paths nor images provided, skipping sky segmentation"
        )
        return None

    resolved_model = _ensure_skyseg_model(skyseg_model_path)
    model_fingerprint = _get_model_fingerprint(resolved_model)
    skyseg_session = _create_ort_session(resolved_model)
    sky_masks = []

    if sky_mask_visualization_dir is not None:
        os.makedirs(sky_mask_visualization_dir, exist_ok=True)

    if images is not None:
        if image_paths is None and image_folder is not None:
            image_paths = _list_image_files(image_folder)

        num_images = images.shape[0]
        if num_frames is not None:
            num_images = min(num_images, num_frames)
        if image_paths is not None:
            image_paths = image_paths[:num_images]

        if sky_mask_dir is None and image_folder is not None:
            sky_mask_dir = image_folder.rstrip("/") + "_sky_masks"
        refresh_cache = _prepare_sky_mask_cache(sky_mask_dir)

        for i in range(num_images):
            image_rgb = _image_to_rgb_uint8(images[i])
            image_h, image_w = image_rgb.shape[:2]
            image_name = _get_mask_filename(
                image_paths,
                i,
                image_rgb=image_rgb,
                model_fingerprint=model_fingerprint,
            )
            mask_filepath = (
                os.path.join(sky_mask_dir, image_name)
                if sky_mask_dir is not None
                else None
            )

            if (
                mask_filepath is not None
                and not refresh_cache
                and os.path.exists(mask_filepath)
            ):
                sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                if sky_mask is None or sky_mask.shape[:2] != (image_h, image_w):
                    sky_mask = segment_sky_from_array(
                        image_rgb, skyseg_session, image_h, image_w
                    )
                    if not cv2.imwrite(mask_filepath, _mask_to_uint8(sky_mask)):
                        raise OSError(f"Failed to write sky mask: {mask_filepath}")
            else:
                sky_mask = segment_sky_from_array(
                    image_rgb, skyseg_session, image_h, image_w
                )
                if mask_filepath is not None:
                    if not cv2.imwrite(mask_filepath, _mask_to_uint8(sky_mask)):
                        raise OSError(f"Failed to write sky mask: {mask_filepath}")

            if sky_mask_visualization_dir is not None:
                vis_name = _get_visualization_filename(image_paths, i)
                _save_sky_mask_visualization(
                    image_rgb,
                    sky_mask,
                    os.path.join(sky_mask_visualization_dir, vis_name),
                )

            if target_shape is not None and sky_mask.shape[:2] != target_shape:
                sky_mask = cv2.resize(
                    sky_mask,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            sky_masks.append(_mask_to_float(sky_mask))
    else:
        if image_paths is None and image_folder is not None:
            image_paths = _list_image_files(image_folder)

    if images is None and image_paths is not None:
        if len(image_paths) == 0:
            return None
        if num_frames is not None:
            image_paths = image_paths[:num_frames]
        if sky_mask_dir is None:
            if image_folder is None:
                image_folder = os.path.dirname(image_paths[0])
            sky_mask_dir = image_folder.rstrip("/") + "_sky_masks"
        refresh_cache = _prepare_sky_mask_cache(sky_mask_dir)

        for i, image_path in enumerate(image_paths):
            image_name = _get_mask_filename(
                image_paths, i, model_fingerprint=model_fingerprint
            )
            mask_filepath = os.path.join(sky_mask_dir, image_name)

            if not refresh_cache and os.path.exists(mask_filepath):
                sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                if sky_mask is None:
                    sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)
            else:
                sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

            if sky_mask is None:
                continue

            if sky_mask_visualization_dir is not None:
                image_bgr = cv2.imread(image_path)
                if image_bgr is not None:
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    vis_name = _get_visualization_filename(image_paths, i)
                    _save_sky_mask_visualization(
                        image_rgb,
                        sky_mask,
                        os.path.join(sky_mask_visualization_dir, vis_name),
                    )

            if target_shape is not None and sky_mask.shape[:2] != target_shape:
                sky_mask = cv2.resize(
                    sky_mask,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            sky_masks.append(_mask_to_float(sky_mask))

    if not sky_masks:
        return None
    try:
        return np.stack(sky_masks, axis=0)
    except ValueError:
        return np.array(sky_masks, dtype=object)


def apply_sky_segmentation(
    conf: np.ndarray,
    image_folder: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    images: Optional[np.ndarray] = None,
    skyseg_model_path: Optional[str] = None,
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
) -> np.ndarray:
    """Apply sky segmentation to a (S, H, W) confidence array."""
    num_frames, height, width = conf.shape

    sky_mask_array = load_or_create_sky_masks(
        image_folder=image_folder,
        image_paths=image_paths,
        images=images,
        skyseg_model_path=skyseg_model_path,
        sky_mask_dir=sky_mask_dir,
        sky_mask_visualization_dir=sky_mask_visualization_dir,
        target_shape=(height, width),
        num_frames=num_frames,
    )
    if sky_mask_array is None:
        return conf

    if sky_mask_array.shape[0] < num_frames:
        padded = np.zeros((num_frames, height, width), dtype=sky_mask_array.dtype)
        padded[: sky_mask_array.shape[0]] = sky_mask_array
        sky_mask_array = padded
    elif sky_mask_array.shape[0] > num_frames:
        sky_mask_array = sky_mask_array[:num_frames]

    sky_mask_binary = (sky_mask_array > _SKYSEG_SOFT_THRESHOLD).astype(np.float32)
    return conf * sky_mask_binary
