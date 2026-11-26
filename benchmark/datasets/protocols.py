# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Online evaluation protocols with patch16 input geometry."""

from importlib.resources import files
import json
from lingbot_map2.utils.load_fn import uniform_keyframe_ids

SCENES = json.loads(files("benchmark").joinpath("configs/scenes.json").read_text())
FRAME_COUNTS = json.loads(
    files("benchmark").joinpath("configs/frame_counts.json").read_text()
)
PROTOCOLS = {}
for dataset, kind, wh, stride in (
    ("kitti", "trajectory", [512, 288], 1),
    ("vbr", "trajectory", [512, 288], 1),
    ("oxford", "trajectory", [512, 384], 1),
    ("crocodl", "pose", [512, 384], 5),
    ("tat", "pose", [512, 288], 1),
    ("princeton365", "pose", [512, 272], 10),
    ("crocodl", "dense", [512, 384], 5),
    ("euroc", "dense", [512, 320], 5),
    ("oxford", "dense", [512, 384], 10),
):
    name = f"{dataset}_{kind}"
    PROTOCOLS[name] = dict(
        name=name,
        version=3 if kind == "trajectory" else 2,
        dataset=dataset,
        kind=kind,
        input_wh=wh,
        stride=stride,
        prefix=8,
        window=64,
        input_scope="full_stream" if kind == "trajectory" else "selected_views",
        persistence="prefix8_then_ceil_N_over_320"
        if kind == "trajectory"
        else "all_selected",
        preprocessing="float32_area_cover_center_crop",
        expected_scenes=SCENES[dataset],
        expected_raw_counts=FRAME_COUNTS[name],
    )
    if kind == "dense":
        PROTOCOLS[name]["geometry"] = dict(
            alignment="camera_centers"
            if dataset == "euroc"
            else "pixel_correspondence",
            grid_wh={"crocodl": [256, 192], "euroc": [256, 160], "oxford": [518, 388]}[
                dataset
            ],
            voxel_m=0.05 if dataset == "oxford" else 0.02,
            icp_threshold_m=0.5 if dataset == "oxford" else 0.1,
            icp_iterations=20 if dataset == "oxford" else 30,
            alignment_max_points=250000 if dataset == "crocodl" else 0,
            f1_threshold_m=4.0 if dataset == "oxford" else 0.25,
            confidence_filter=False,
            voxel_output_dtype="float32" if dataset == "oxford" else "float64",
        )
        if dataset == "euroc":
            PROTOCOLS[name]["reference_caveat"] = (
                "Provisional custom map-to-scan evaluation: camera-to-scan orientation, alignment and rasterization/ICP are not yet validated against the complete reference dataset; not the official EuRoC benchmark."
            )


def get_protocol(name, input_scope=None):
    protocol = dict(PROTOCOLS[name])
    if input_scope is not None:
        if not isinstance(input_scope, str):
            raise ValueError("input_scope must be a string")
        allowed = (
            {"full_stream", "keyframes"}
            if protocol["kind"] == "trajectory"
            else {"selected_views"}
        )
        if input_scope not in allowed:
            raise ValueError(f"{name}: input_scope must be one of {sorted(allowed)}")
        protocol["input_scope"] = input_scope
    return protocol


def selection(protocol, raw_count):
    if type(raw_count) is not int or raw_count < protocol["prefix"]:
        raise ValueError("sequence must contain at least eight raw frames")
    ids = list(range(0, raw_count, protocol["stride"]))
    if len(ids) < protocol["prefix"]:
        raise ValueError("selected sequence is shorter than the initialization prefix")
    if protocol["kind"] == "trajectory":
        keyframes = uniform_keyframe_ids(raw_count, prefix=protocol["prefix"])
        persistent = set(keyframes)
        if protocol["input_scope"] == "keyframes":
            ids = keyframes
    else:
        persistent = set(ids)
    return ids, [i in persistent for i in ids]
