# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Explicit identity and training contract for the causal-only weight release."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointConfig:
    name: str
    sha256: str
    schema_version: int = 1
    training_mode: str = "causal"
    architecture: str = "dinov3_vitl16_causal24"
    spatial_rope: str = "max"
    scale_frames: int = 8
    sliding_window: int = 64
    camera_iterations: int = 4
    max_frames: int = 1024
    width: int = 512
    height: int = 288

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).read_text())
        fields = set(cls.__dataclass_fields__)
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError(f"checkpoint config must contain exactly {sorted(fields)}")
        config = cls(**data)
        if config.schema_version != 1 or type(config.schema_version) is not int:
            raise ValueError("unsupported checkpoint schema")
        if (
            config.training_mode != "causal"
            or config.architecture != "dinov3_vitl16_causal24"
        ):
            raise ValueError(
                "this release accepts only pure causal LingBot-Map2 weights"
            )
        if config.spatial_rope not in {"max", "none"}:
            raise ValueError("unsupported global spatial RoPE contract")
        if not isinstance(config.name, str) or not config.name.strip():
            raise ValueError("checkpoint name must be non-empty")
        if (
            not isinstance(config.sha256, str)
            or len(config.sha256) != 64
            or any(c not in "0123456789abcdef" for c in config.sha256)
            or config.sha256 == "0" * 64
        ):
            raise ValueError(
                "provide the published SHA256; placeholders are not weights"
            )
        for field in (
            "scale_frames",
            "sliding_window",
            "camera_iterations",
            "max_frames",
            "width",
            "height",
        ):
            value = getattr(config, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if (
            config.camera_iterations != 4
            or config.max_frames > 1024
            or config.scale_frames > config.max_frames
        ):
            raise ValueError("require Camera4 and scale_frames <= max_frames <= 1024")
        if config.width % 16 or config.height % 16:
            raise ValueError("input dimensions must be multiples of 16")
        return config

    def to_dict(self):
        return asdict(self)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_paths(checkpoint, config=None):
    """Resolve aliases before looking for the checkpoint's adjacent JSON."""
    path = Path(checkpoint).expanduser().resolve(strict=True)
    config_path = (
        Path(config).expanduser() if config is not None else Path(str(path) + ".json")
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            "provide the published causal checkpoint config with --config "
            "or as <checkpoint>.json; there is no bundled weight preset"
        )
    return path, config_path.resolve()


def resolve_checkpoint(checkpoint, config=None):
    path, config_path = checkpoint_paths(checkpoint, config)
    spec = CheckpointConfig.read(config_path)
    if sha256_file(path) != spec.sha256:
        raise ValueError(
            "checkpoint SHA256 mismatch; use the matching published weights/config pair"
        )
    return path, spec, config_path
