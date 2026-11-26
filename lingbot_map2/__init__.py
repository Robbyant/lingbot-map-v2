# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""LingBot-Map2 Streaming: pure causal online inference."""

__all__ = ["GCRStream", "load_model"]


def __getattr__(name):
    # Data preparation and keyframe selection do not need PyTorch or CUDA.
    if name in __all__:
        from . import models

        value = getattr(models, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
