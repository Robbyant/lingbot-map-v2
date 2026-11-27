# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 The Robbyant Team Authors. All rights reserved.
# Adapted from Robbyant/lingbot-map, lingbot_map/vis/point_cloud_viewer.py
# (b9dd1b1de535e6d36e777fb42b8433c781bfd783). See LICENSES/THIRD_PARTY.md.

"""Interactive Viser 3D/4D point cloud viewer aligned with LingBot-Map V1."""

import json
import logging
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from threading import Condition, RLock, Thread
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
from matplotlib import colormaps
import numpy as np
import viser
import viser.transforms as tf

from .glb_export import (
    _extract_c2w_and_w2c,
    _unproject_depth_to_world,
    apply_scene_alignment,
    build_trajectory_tube,
    integrate_camera_into_scene,
)
from .sky_segmentation import apply_sky_segmentation
from .utils import CameraState

logger = logging.getLogger(__name__)


def _find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg from PATH or imageio-ffmpeg when installed."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _set_frustum_thickness(handle, thickness: float) -> None:
    if hasattr(handle, "thickness"):
        handle.thickness = thickness
    elif hasattr(handle, "line_width"):
        handle.line_width = thickness


def _set_glb_mesh_alpha_mode(
    glb_bytes: bytes, mesh_name: str, alpha_mode: str = "BLEND"
) -> bytes:
    """Set glTF 2.0 material alphaMode on primitives of `mesh_name` inside a GLB."""
    if len(glb_bytes) < 20:
        return glb_bytes
    magic, version, _ = struct.unpack("<4sII", glb_bytes[:12])
    chunk_len, chunk_type = struct.unpack("<I4s", glb_bytes[12:20])
    if magic != b"glTF" or version != 2 or chunk_type != b"JSON":
        return glb_bytes

    json_end = 20 + chunk_len
    tree = json.loads(glb_bytes[20:json_end].decode("utf-8"))
    meshes = tree.get("meshes") or []
    materials = tree.setdefault("materials", [])

    # Count material references so shared materials are cloned rather than mutated globally.
    ref_counts: Dict[int, int] = {}
    for mesh in meshes:
        for prim in mesh.get("primitives") or []:
            mat_idx = prim.get("material")
            if isinstance(mat_idx, int):
                ref_counts[mat_idx] = ref_counts.get(mat_idx, 0) + 1

    modified = False
    for mesh in meshes:
        if mesh.get("name") != mesh_name:
            continue
        for prim in mesh.get("primitives") or []:
            mat_idx = prim.get("material")
            if isinstance(mat_idx, int) and 0 <= mat_idx < len(materials):
                if ref_counts.get(mat_idx, 0) <= 1:
                    materials[mat_idx]["alphaMode"] = alpha_mode
                else:
                    cloned = json.loads(json.dumps(materials[mat_idx]))
                    cloned["alphaMode"] = alpha_mode
                    prim["material"] = len(materials)
                    materials.append(cloned)
                    ref_counts[mat_idx] -= 1
            else:
                prim["material"] = len(materials)
                materials.append(
                    {
                        "pbrMetallicRoughness": {
                            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                            "metallicFactor": 0.0,
                            "roughnessFactor": 1.0,
                        },
                        "alphaMode": alpha_mode,
                    }
                )
            modified = True

    if not modified:
        return glb_bytes

    json_bytes = json.dumps(tree, separators=(",", ":")).encode("utf-8")
    pad = (-len(json_bytes)) % 4
    if pad:
        json_bytes += b" " * pad
    rest = glb_bytes[json_end:]
    total_len = 12 + 8 + len(json_bytes) + len(rest)
    return (
        struct.pack("<4sII", b"glTF", 2, total_len)
        + struct.pack("<I4s", len(json_bytes), b"JSON")
        + json_bytes
        + rest
    )


class _DebouncedTask:
    """Coalesce rapid GUI slider events onto a single background worker thread."""

    def __init__(
        self,
        name: str,
        fn,
        delay: float,
        closed_fn,
        on_error=None,
        on_success=None,
    ):
        self._name = name
        self._fn = fn
        self._delay = float(delay)
        self._closed_fn = closed_fn
        self._on_error = on_error
        self._on_success = on_success
        self._cond = Condition()
        self._gen = 0
        self._done_gen = 0
        self._deadline = 0.0
        self._stopped = False
        self.last_error: Optional[Exception] = None
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def trigger(self) -> None:
        with self._cond:
            if self._stopped:
                return
            self._gen += 1
            self._deadline = time.monotonic() + self._delay
            self._cond.notify()

    @property
    def pending(self) -> bool:
        with self._cond:
            return self._gen != self._done_gen

    def is_stale(self, gen: int) -> bool:
        with self._cond:
            return self._stopped or self._gen != gen

    def cancel_pending(self) -> None:
        with self._cond:
            self._gen += 1
            self._done_gen = self._gen
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._stopped and self._gen == self._done_gen:
                    self._cond.wait()
                if self._stopped or self._closed_fn():
                    return
                remaining = self._deadline - time.monotonic()
                if remaining > 0:
                    self._cond.wait(timeout=remaining)
                    continue
                target_gen = self._gen
            if self._stopped or self._closed_fn():
                return
            try:
                self._fn(target_gen)
            except Exception as error:
                if not self.is_stale(target_gen):
                    self.last_error = error
                    if self._on_error is not None:
                        try:
                            self._on_error(self._name, error)
                        except Exception:
                            logger.exception("Failed to report error for %s", self._name)
            else:
                if not self.is_stale(target_gen):
                    self.last_error = None
                    if self._on_success is not None:
                        try:
                            self._on_success(self._name)
                        except Exception:
                            logger.exception("Failed to clear error for %s", self._name)
            with self._cond:
                if target_gen > self._done_gen:
                    self._done_gen = target_gen


class PointCloudViewer:
    """Interactive V1-aligned Viser point cloud viewer.

    Consumes completed Online streaming predictions (`pred_dict` or `pc_list`)
    and exposes the full V1 interactive interface:
    - Current RGB frame preview (`Video Display`)
    - Up-direction reset and preset view directions (`Reset View Direction`)
    - 3D cumulative / 4D single-frame mode toggle
    - Point size, absolute visibility threshold (1..5, default 1.5), point
      downsampling, camera size, camera visibility and camera downsampling
    - Screenshot capture, GLB export (Points / Spheres + cameras + trajectory)
      and FFmpeg video export (point cloud + optional original RGB video)
    - Timeline scrubbing and playback controls (`Playback`)
    """

    def __init__(
        self,
        *,
        pc_list=None,
        color_list=None,
        conf_list=None,
        cam_dict=None,
        image_mask=None,
        edge_color_list=None,
        host: str = "0.0.0.0",
        port: int = 8080,
        show_camera: bool = True,
        vis_threshold: float = 1.5,
        conf_threshold: Optional[float] = None,
        size: int = 512,
        downsample_factor: int = 10,
        point_size: float = 0.00001,
        camera_size: float = 0.01,
        pred_dict: Optional[Dict[str, Any]] = None,
        use_point_map: bool = False,
        mask_sky: bool = False,
        image_folder: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        skyseg_model_path: Optional[str] = None,
        sky_mask_dir: Optional[str] = None,
        sky_mask_visualization_dir: Optional[str] = None,
        depth_stride: int = 1,
        label: str = "LingBot-Map 2.0",
    ):
        if conf_threshold is not None:
            vis_threshold = conf_threshold
        if depth_stride < 1:
            raise ValueError("depth_stride must be positive")
        if not np.isfinite(point_size) or point_size <= 0:
            raise ValueError("point_size must be finite and positive")
        if not np.isfinite(camera_size) or camera_size <= 0:
            raise ValueError("camera_size must be finite and positive")
        if not 1.0 <= vis_threshold <= 5.0 or downsample_factor < 1:
            raise ValueError(
                "visibility threshold must be in [1, 5] and downsample factor positive"
            )
        if pred_dict is None and pc_list is None:
            raise ValueError("provide pred_dict or pc_list")

        self.size = size
        self.vis_threshold = float(vis_threshold)
        self.point_size = float(point_size)
        self.camera_size = float(camera_size)
        self.downsample_factor = int(downsample_factor)
        self.show_camera = bool(show_camera)
        self.image_mask = image_mask

        self._default_vis_threshold = self.vis_threshold
        self._default_point_size = self.point_size
        self._default_camera_size = self.camera_size
        self._default_downsample_factor = self.downsample_factor
        self._default_show_camera = self.show_camera
        self._default_camera_downsample_factor = 1
        self._default_focal = 533.0
        self._default_framerate = 20.0

        self._lock = RLock()
        self._closed = False
        self._updating_timestep_from_loop = False
        self._view: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._last_displayed_image_idx: Optional[int] = None

        self.on_replay = False
        self._saving_video = False
        self._rebuilding = False
        self._export_timestep: Optional[int] = None
        self._client_telemetry: Dict[int, Dict[str, Any]] = {}
        self._telemetry_last_sync: Dict[int, float] = {}
        self._pc_base_cache: Dict[
            int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        ] = {}
        self.last_display_error: Optional[str] = None
        self.fourd = False
        self.original_images: List[np.ndarray] = []
        self.frame_nodes: List[Any] = []
        self.pc_handles: List[Any] = []
        self.cam_handles: List[Any] = []
        self.cam_axes_handles: List[Any] = []

        if pred_dict is not None:
            pc_list, color_list, conf_list, cam_dict = self._process_pred_dict(
                pred_dict,
                use_point_map=use_point_map,
                mask_sky=mask_sky,
                image_folder=image_folder,
                image_paths=image_paths,
                skyseg_model_path=skyseg_model_path,
                sky_mask_dir=sky_mask_dir,
                sky_mask_visualization_dir=sky_mask_visualization_dir,
                depth_stride=depth_stride,
            )

        self.pcs, self.all_steps = self.read_data(
            pc_list, color_list, conf_list, edge_color_list
        )
        self.cam_dict = cam_dict
        self.num_frames = len(self.all_steps)

        if not self.original_images and color_list:
            for col in color_list:
                arr = np.asarray(col)
                if arr.ndim == 3 and arr.shape[-1] == 3 and arr.size > 0:
                    if arr.dtype != np.uint8:
                        scale = 255.0 if arr.max() <= 1.0 else 1.0
                        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
                    self.original_images.append(arr)

        self.server = viser.ViserServer(host=host, port=port, label=label)
        self.server.gui.configure_theme(
            titlebar_content=None, control_layout="collapsible"
        )
        self.server.scene.set_up_direction("-y")

        self._pc_rebuilder = _DebouncedTask(
            "Point Cloud Update",
            lambda gen: self._regenerate_point_clouds(
                cancel_check=lambda: self._pc_rebuilder.is_stale(gen)
            ),
            delay=0.18,
            closed_fn=lambda: self._closed,
            on_error=self._report_display_error,
            on_success=self._clear_display_error,
        )
        self._cam_rebuilder = _DebouncedTask(
            "Camera Update",
            lambda gen: self._regenerate_cameras(
                cancel_check=lambda: self._cam_rebuilder.is_stale(gen)
            ),
            delay=0.15,
            closed_fn=lambda: self._closed,
            on_error=self._report_display_error,
            on_success=self._clear_display_error,
        )
        self._psize_updater = _DebouncedTask(
            "Point Size Update",
            lambda gen: self._apply_point_size(
                cancel_check=lambda: self._psize_updater.is_stale(gen)
            ),
            delay=0.05,
            closed_fn=lambda: self._closed,
            on_error=self._report_display_error,
            on_success=self._clear_display_error,
        )
        self._camsize_updater = _DebouncedTask(
            "Camera Size Update",
            lambda gen: self._apply_camera_size(
                cancel_check=lambda: self._camsize_updater.is_stale(gen)
            ),
            delay=0.05,
            closed_fn=lambda: self._closed,
            on_error=self._report_display_error,
            on_success=self._clear_display_error,
        )
        self._focal_updater = _DebouncedTask(
            "Focal Length Update",
            lambda gen: self._apply_focal_length(
                cancel_check=lambda: self._focal_updater.is_stale(gen)
            ),
            delay=0.04,
            closed_fn=lambda: self._closed,
            on_error=self._report_display_error,
            on_success=self._clear_display_error,
        )
        self._debounced_tasks = (
            self._pc_rebuilder,
            self._cam_rebuilder,
            self._psize_updater,
            self._camsize_updater,
            self._focal_updater,
        )

        self._setup_gui()
        self._setup_playback_gui()
        self._build_scene()
        self.fit_view(smooth=False)
        self.server.on_client_connect(self._connect_client)
        if hasattr(self.server, "on_client_disconnect"):
            self.server.on_client_disconnect(self._disconnect_client)

    def _process_pred_dict(
        self,
        pred_dict: Dict[str, Any],
        use_point_map: bool,
        mask_sky: bool,
        image_folder: Optional[str],
        image_paths: Optional[List[str]] = None,
        sky_mask_dir: Optional[str] = None,
        sky_mask_visualization_dir: Optional[str] = None,
        depth_stride: int = 1,
        skyseg_model_path: Optional[str] = None,
    ) -> Tuple[List, List, List, Dict]:
        """Extract per-frame point clouds, colors, confidences and camera poses."""
        images = np.asarray(pred_dict["images"])
        c2w_all, _ = _extract_c2w_and_w2c(pred_dict)
        intrinsics_cam = np.asarray(
            pred_dict.get("intrinsics", pred_dict.get("intrinsic"))
        )

        if images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
            colors = images.transpose(0, 2, 3, 1)
        else:
            colors = images

        num_frames = len(c2w_all)
        self.original_images = []
        for i in range(num_frames):
            img = colors[i]
            if img.dtype != np.uint8:
                scale = 255.0 if img.size > 0 and img.max() <= 1.0 else 1.0
                img = np.clip(img * scale, 0, 255).astype(np.uint8)
            self.original_images.append(img)

        selected = list(range(0, num_frames, depth_stride))
        sel_index_map = {frame_idx: pos for pos, frame_idx in enumerate(selected)}

        depth_map = pred_dict.get("depth")
        depth_conf = pred_dict.get("depth_conf")

        if selected:
            if not use_point_map or pred_dict.get("world_points") is None:
                sel_depth = np.asarray(depth_map)[selected]
                sel_world_points = _unproject_depth_to_world(
                    sel_depth, c2w_all[selected], intrinsics_cam[selected]
                )
                raw_conf = depth_conf
            else:
                sel_world_points = np.ascontiguousarray(
                    np.asarray(pred_dict["world_points"])[selected]
                )
                raw_conf = pred_dict.get("world_points_conf", depth_conf)

            sel_conf = (
                np.ascontiguousarray(np.asarray(raw_conf, dtype=np.float32)[selected])
                if raw_conf is not None
                else None
            )

            if mask_sky and sel_conf is not None:
                sel_paths = (
                    [image_paths[i] for i in selected if i < len(image_paths)]
                    if image_paths is not None
                    else None
                )
                sel_conf = apply_sky_segmentation(
                    sel_conf,
                    image_folder=image_folder,
                    image_paths=sel_paths,
                    images=images[selected],
                    skyseg_model_path=skyseg_model_path,
                    sky_mask_dir=sky_mask_dir,
                    sky_mask_visualization_dir=sky_mask_visualization_dir,
                )
        else:
            sel_world_points = np.zeros((0, 0, 0, 3), dtype=np.float32)
            sel_conf = None

        pc_list, color_list, conf_list = [], [], []
        for i in range(num_frames):
            pos = sel_index_map.get(i)
            if pos is None:
                pc_list.append(np.zeros((0, 0, 3), dtype=np.float32))
                color_list.append(np.zeros((0, 0, 3), dtype=np.uint8))
                conf_list.append(np.zeros((0, 0), dtype=np.float32))
            else:
                pc_list.append(sel_world_points[pos])
                color_list.append(self.original_images[i])
                if sel_conf is not None:
                    conf_list.append(sel_conf[pos])
                else:
                    conf_list.append(
                        np.ones(sel_world_points[pos].shape[:2], dtype=np.float32)
                    )

        cam_dict = {
            "focal": [float(intrinsics_cam[i, 0, 0]) for i in range(num_frames)],
            "fy": [float(intrinsics_cam[i, 1, 1]) for i in range(num_frames)],
            "pp": [
                (float(intrinsics_cam[i, 0, 2]), float(intrinsics_cam[i, 1, 2]))
                for i in range(num_frames)
            ],
            "image_size": [
                (
                    int(self.original_images[i].shape[0]),
                    int(self.original_images[i].shape[1]),
                )
                for i in range(num_frames)
            ],
            "R": [c2w_all[i, :3, :3] for i in range(num_frames)],
            "t": [c2w_all[i, :3, 3] for i in range(num_frames)],
        }
        return pc_list, color_list, conf_list, cam_dict

    def _compute_scene_center_and_scale(self) -> Tuple[np.ndarray, float]:
        """Compute scene center and scale from camera positions (or points fallback)."""
        if self.cam_dict is not None and "t" in self.cam_dict and self.all_steps:
            cam_positions = np.array(
                [self.cam_dict["t"][s] for s in self.all_steps], dtype=np.float64
            )
            center = np.mean(cam_positions, axis=0)
            scale = (
                float(np.linalg.norm(np.ptp(cam_positions, axis=0)))
                if len(cam_positions) > 1
                else 1.0
            )
            return center, max(scale, 0.1)

        all_pts = []
        for step in self.all_steps:
            pc = np.asarray(self.pcs[step]["pc"]).reshape(-1, 3)
            if len(pc) > 1000:
                pc = pc[:: len(pc) // 1000]
            if len(pc) > 0:
                all_pts.append(pc)
        if not all_pts:
            return np.zeros(3, dtype=np.float64), 1.0
        stacked = np.concatenate(all_pts, axis=0)
        center = np.median(stacked, axis=0)
        extent = np.percentile(stacked, 95, axis=0) - np.percentile(stacked, 5, axis=0)
        return center, max(float(np.linalg.norm(extent)), 0.1)

    def _set_view(self, client: viser.ClientHandle) -> None:
        if self._view is not None:
            position, target, up = self._view
            client.camera.up_direction = tuple(up)
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(target)

    def fit_view(self, smooth: bool = False) -> None:
        """Reset clients to the V1 3/4 overview framed by camera centers."""
        direction = np.array([0.5, -0.6, 0.6], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        self._reset_view_to_direction(direction, smooth=smooth)

    def reset_defaults(self) -> None:
        """Restore display sliders, 3D mode, playback settings, and initial view."""
        if self._saving_video:
            return
        for task in self._debounced_tasks:
            task.cancel_pending()
        with self._lock:
            need_rebuild_pc = (
                abs(self.vis_threshold - self._default_vis_threshold) > 1e-12
                or int(self.downsample_slider.value) != self._default_downsample_factor
            )
            need_rebuild_cam = (
                self.show_camera != self._default_show_camera
                or int(self.camera_downsample_slider.value)
                != self._default_camera_downsample_factor
            )

            self.fourd = False
            self.vis_threshold = self._default_vis_threshold
            self.point_size = self._default_point_size
            self.camera_size = self._default_camera_size
            self.downsample_factor = self._default_downsample_factor
            self.show_camera = self._default_show_camera

            self.focal_slider.value = self._default_focal
            self.psize_slider.value = self._default_point_size
            self.camsize_slider.value = self._default_camera_size
            self.downsample_slider.value = self._default_downsample_factor
            self.show_camera_checkbox.value = self._default_show_camera
            self.vis_threshold_slider.value = self._default_vis_threshold
            self.camera_downsample_slider.value = self._default_camera_downsample_factor

            self.show_video_checkbox.value = True
            if self.current_frame_image is not None:
                self.current_frame_image.visible = True

            self.gui_framerate.value = self._default_framerate
            self.gui_playing.value = True

            self.glb_show_cam_checkbox.value = True
            self.glb_cam_scale_slider.value = 1.0
            self.glb_frustum_thickness_slider.value = 3.0
            self.glb_trajectory_checkbox.value = True
            self.glb_trajectory_radius_slider.value = 0.005
            self.glb_mode_dropdown.value = "Points"
            self.glb_sphere_radius_slider.value = 0.005
            self.glb_sphere_radius_slider.disabled = True
            self.glb_max_sphere_pts_slider.value = 100000
            self.glb_max_sphere_pts_slider.disabled = True
            self.glb_opacity_slider.value = 1.0
            self.glb_saturation_slider.value = 1.0
            self.glb_brightness_slider.value = 1.0

            self.screenshot_resolution.value = "1920x1080"
            self.video_save_fps.value = 30
            self.video_resolution.value = "1920x1080"
            self.save_original_video_checkbox.value = True

            if need_rebuild_pc:
                self._regenerate_point_clouds()
            else:
                self._apply_point_size(flush=False)

            if need_rebuild_cam:
                if self.show_camera:
                    self._regenerate_cameras()
                else:
                    with self.server.atomic():
                        for handle in self.cam_handles:
                            handle.visible = False
                        for axes in self.cam_axes_handles:
                            axes.visible = False
            else:
                self._apply_camera_size(flush=False)

            self.update_frame_visibility()
            self._update_current_frame_image(int(self.gui_timestep.value))

        for task in self._debounced_tasks:
            task.last_error = None
        self._clear_display_error()
        self.fit_view(smooth=False)
        default_vfov = self._focal_to_vfov(self._default_focal)
        for client in self.server.get_clients().values():
            client.camera.fov = default_vfov
            self._sync_client_telemetry(client, force=True)
        self.server.flush()

    def _reset_view_to_direction(
        self,
        direction: np.ndarray,
        up: np.ndarray = np.array([0.0, -1.0, 0.0]),
        distance_scale: float = 1.5,
        smooth: bool = True,
    ) -> None:
        """Reset viewer camera to look at scene center from a given direction."""
        center, scale = self._compute_scene_center_and_scale()
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        up = np.asarray(up, dtype=np.float64)
        position = center + direction * (scale * distance_scale)
        self._view = (position, center, up)

        for client in self.server.get_clients().values():
            if smooth:
                self._smooth_camera_transition(
                    client,
                    target_position=position,
                    target_look_at=center,
                    target_up=up,
                    duration=0.4,
                )
            else:
                client.camera.up_direction = tuple(up)
                client.camera.position = tuple(position)
                client.camera.look_at = tuple(center)

    def _setup_gui(self) -> None:
        """Setup V1-aligned GUI controls in the exact V1 folder order."""
        gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            if client is None:
                return
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )

        self.btn_reset_defaults = self.server.gui.add_button(
            "Reset Defaults",
            hint="Reset display sliders, 3D mode, playback FPS, and view to initial defaults.",
        )

        @self.btn_reset_defaults.on_click
        def _(_) -> None:
            self.reset_defaults()

        # Video frame display controls — kept at top so the current frame is always visible
        with self.server.gui.add_folder("Video Display"):
            self.show_video_checkbox = self.server.gui.add_checkbox(
                "Show Current Frame", initial_value=True
            )
            if self.original_images:
                self.current_frame_image = self.server.gui.add_image(
                    self.original_images[0],
                    label="Current Frame",
                    format="jpeg",
                    jpeg_quality=80,
                )
                self._last_displayed_image_idx = 0
            else:
                self.current_frame_image = None

        # Preset view direction buttons
        with self.server.gui.add_folder("Reset View Direction"):
            btn_look_at_center = self.server.gui.add_button(
                "Look At Scene Center",
                hint="Reset orbit center to the scene center (fixes orbit after dragging).",
            )
            btn_overview = self.server.gui.add_button(
                "Overview",
                hint="Reset to a 3/4 overview of the scene.",
            )
            btn_front = self.server.gui.add_button(
                "Front (+Z)",
                hint="View scene from the front.",
            )
            btn_back = self.server.gui.add_button(
                "Back (-Z)",
                hint="View scene from the back.",
            )
            btn_top = self.server.gui.add_button(
                "Top (-Y)",
                hint="View scene from above (bird's eye).",
            )
            btn_left = self.server.gui.add_button(
                "Left (-X)",
                hint="View scene from the left.",
            )
            btn_right = self.server.gui.add_button(
                "Right (+X)",
                hint="View scene from the right.",
            )
            btn_first_cam = self.server.gui.add_button(
                "First Camera",
                hint="Reset to the first camera's viewpoint.",
            )

        @btn_look_at_center.on_click
        def _(_) -> None:
            center, _ = self._compute_scene_center_and_scale()
            for client in self.server.get_clients().values():
                client.camera.look_at = tuple(center)

        @btn_overview.on_click
        def _(_) -> None:
            d = np.array([0.5, -0.6, 0.6])
            self._reset_view_to_direction(d / np.linalg.norm(d))

        @btn_front.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([0.0, 0.0, 1.0]))

        @btn_back.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([0.0, 0.0, -1.0]))

        @btn_top.on_click
        def _(_) -> None:
            self._reset_view_to_direction(
                np.array([0.0, -1.0, 0.0]),
                up=np.array([0.0, 0.0, 1.0]),
            )

        @btn_left.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([-1.0, 0.0, 0.0]))

        @btn_right.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([1.0, 0.0, 0.0]))

        @btn_first_cam.on_click
        def _(_) -> None:
            self._move_to_camera(0, smooth=True)

        # 3D / 4D mode controls
        self.btn_4d = self.server.gui.add_button("4D (Only Show Current Frame)")
        self.btn_3d = self.server.gui.add_button("3D (Show All Frames)")

        @self.btn_4d.on_click
        def _(_) -> None:
            if self._saving_video:
                return
            self.fourd = True
            self.update_frame_visibility()
            self.server.flush()

        @self.btn_3d.on_click
        def _(_) -> None:
            if self._saving_video:
                return
            self.fourd = False
            self.update_frame_visibility()
            self.server.flush()

        # Main display sliders & checkboxes
        self.focal_slider = self.server.gui.add_slider(
            "Focal Length", min=50, max=2000, step=1, initial_value=533
        )
        self.psize_slider = self.server.gui.add_slider(
            "Point Size",
            min=0.00001,
            max=0.1,
            step=0.00001,
            initial_value=self.point_size,
        )
        self.camsize_slider = self.server.gui.add_slider(
            "Camera Size",
            min=0.001,
            max=0.5,
            step=0.001,
            initial_value=self.camera_size,
        )
        self.downsample_slider = self.server.gui.add_slider(
            "Downsample Factor",
            min=1,
            max=200,
            step=1,
            initial_value=self.downsample_factor,
        )
        self.show_camera_checkbox = self.server.gui.add_checkbox(
            "Show Camera", initial_value=self.show_camera
        )
        self.vis_threshold_slider = self.server.gui.add_slider(
            "Visibility Threshold",
            min=1.0,
            max=5.0,
            step=0.01,
            initial_value=self.vis_threshold,
        )
        self.camera_downsample_slider = self.server.gui.add_slider(
            "Camera Downsample Factor", min=1, max=50, step=1, initial_value=1
        )
        self.display_status = self.server.gui.add_text(
            "Display Status",
            initial_value="Ready",
            disabled=True,
            visible=False,
        )

        # Screenshot controls
        with self.server.gui.add_folder("Screenshot"):
            self.screenshot_button = self.server.gui.add_button("Take Screenshot")
            self.screenshot_resolution = self.server.gui.add_dropdown(
                "Resolution",
                options=["1920x1080", "2560x1440", "3840x2160", "Current"],
                initial_value="1920x1080",
            )
            self.screenshot_path = self.server.gui.add_text(
                "Save Path", initial_value="screenshot.png"
            )
            self.screenshot_status = self.server.gui.add_text(
                "Status", initial_value="Ready"
            )

        @self.screenshot_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self._take_screenshot(event.client)

        # GLB export controls
        with self.server.gui.add_folder("Export GLB"):
            self.glb_output_path = self.server.gui.add_text(
                "Output Path", initial_value="export.glb"
            )
            self.glb_show_cam_checkbox = self.server.gui.add_checkbox(
                "Include Cameras", initial_value=True
            )
            self.glb_cam_scale_slider = self.server.gui.add_slider(
                "Camera Scale",
                min=0.01,
                max=5.0,
                step=0.01,
                initial_value=1.0,
                hint="Scale factor for camera size in GLB.",
            )
            self.glb_frustum_thickness_slider = self.server.gui.add_slider(
                "Frustum Thickness",
                min=1.0,
                max=10.0,
                step=0.5,
                initial_value=3.0,
                hint="Thickness multiplier for camera frustum edges.",
            )
            self.glb_trajectory_checkbox = self.server.gui.add_checkbox(
                "Show Trajectory",
                initial_value=True,
                hint="Connect cameras with a trajectory line.",
            )
            self.glb_trajectory_radius_slider = self.server.gui.add_slider(
                "Trajectory Radius",
                min=0.001,
                max=0.05,
                step=0.001,
                initial_value=0.005,
                hint="Radius of the trajectory tube.",
            )
            self.glb_mode_dropdown = self.server.gui.add_dropdown(
                "Export Mode",
                options=["Points", "Spheres"],
                initial_value="Points",
                hint="Points: raw (fast). Spheres: each point becomes a small sphere (prettier, slower).",
            )
            self.glb_sphere_radius_slider = self.server.gui.add_slider(
                "Sphere Radius",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=0.005,
                hint="Radius of each sphere in Spheres mode.",
                disabled=True,
            )
            self.glb_max_sphere_pts_slider = self.server.gui.add_slider(
                "Max Sphere Points",
                min=10000,
                max=500000,
                step=10000,
                initial_value=100000,
                hint="Cap point count for Spheres mode to keep file size manageable.",
                disabled=True,
            )
            self.glb_opacity_slider = self.server.gui.add_slider(
                "Opacity",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=1.0,
                hint="Point/sphere opacity (alpha). <1.0 = semi-transparent.",
            )
            self.glb_saturation_slider = self.server.gui.add_slider(
                "Saturation Boost",
                min=0.0,
                max=2.0,
                step=0.1,
                initial_value=1.0,
                hint="Color saturation multiplier. >1 = more vivid, <1 = washed out.",
            )
            self.glb_brightness_slider = self.server.gui.add_slider(
                "Brightness Boost",
                min=0.5,
                max=2.0,
                step=0.1,
                initial_value=1.0,
                hint="Color brightness multiplier.",
            )
            self.glb_export_button = self.server.gui.add_button(
                "Export GLB",
                hint="Export current filtered point clouds and cameras as GLB.",
            )
            self.glb_status = self.server.gui.add_text("Status", initial_value="Ready")

        @self.glb_mode_dropdown.on_update
        def _(_) -> None:
            is_sphere = self.glb_mode_dropdown.value == "Spheres"
            self.glb_sphere_radius_slider.disabled = not is_sphere
            self.glb_max_sphere_pts_slider.disabled = not is_sphere

        @self.glb_export_button.on_click
        def _(_) -> None:
            self._export_glb()

        # Video saving controls
        with self.server.gui.add_folder("Video Saving"):
            self.save_video_button = self.server.gui.add_button(
                "Save Video", disabled=False
            )
            self.video_output_path = self.server.gui.add_text(
                "Output Path", initial_value="output_pointcloud.mp4"
            )
            self.video_save_fps = self.server.gui.add_slider(
                "Video FPS", min=10, max=60, step=1, initial_value=30
            )
            self.video_resolution = self.server.gui.add_dropdown(
                "Resolution",
                options=["1920x1080", "1280x720", "3840x2160"],
                initial_value="1920x1080",
            )
            self.save_original_video_checkbox = self.server.gui.add_checkbox(
                "Also Save Original Video", initial_value=True
            )
            self.video_status = self.server.gui.add_text(
                "Status", initial_value="Ready to save"
            )

        @self.save_video_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self.save_video(
                output_path=self.video_output_path.value,
                fps=int(self.video_save_fps.value),
                resolution=self.video_resolution.value,
                save_original_video=self.save_original_video_checkbox.value,
                client=event.client,
            )

        @self.show_video_checkbox.on_update
        def _(_) -> None:
            if self.current_frame_image is not None:
                self.current_frame_image.visible = self.show_video_checkbox.value
                if self.show_video_checkbox.value:
                    self._update_current_frame_image(int(self.gui_timestep.value))

        @self.focal_slider.on_update
        def _(_) -> None:
            self._focal_updater.trigger()

        @self.psize_slider.on_update
        def _(_) -> None:
            self._psize_updater.trigger()

        @self.camsize_slider.on_update
        def _(_) -> None:
            self._camsize_updater.trigger()

        @self.downsample_slider.on_update
        def _(_) -> None:
            self._pc_rebuilder.trigger()

        @self.show_camera_checkbox.on_update
        def _(_) -> None:
            with self._lock:
                self.show_camera = self.show_camera_checkbox.value
                if self.show_camera:
                    self._cam_rebuilder.cancel_pending()
                    self._regenerate_cameras()
                else:
                    self._cam_rebuilder.cancel_pending()
                    with self.server.atomic():
                        for handle in self.cam_handles:
                            handle.visible = False
                        for axes in self.cam_axes_handles:
                            axes.visible = False
            self.server.flush()

        @self.vis_threshold_slider.on_update
        def _(_) -> None:
            self.vis_threshold = self.vis_threshold_slider.value
            self._pc_rebuilder.trigger()

        @self.camera_downsample_slider.on_update
        def _(_) -> None:
            self._cam_rebuilder.trigger()

    def _setup_playback_gui(self) -> None:
        """Setup V1-aligned Playback folder and timeline callbacks."""
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Train Step",
                min=0,
                max=max(self.num_frames - 1, 0),
                step=1,
                initial_value=0,
                disabled=False,
            )
            self.gui_next_frame = self.server.gui.add_button(
                "Next Step", disabled=False
            )
            self.gui_prev_frame = self.server.gui.add_button(
                "Prev Step", disabled=False
            )
            self.gui_playing = self.server.gui.add_checkbox("Playing", True)
            self.gui_framerate = self.server.gui.add_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=20
            )
            self.gui_framerate_options = self.server.gui.add_button_group(
                "FPS options", ("10", "20", "30", "60")
            )

        @self.gui_next_frame.on_click
        def _(_) -> None:
            if self._saving_video:
                return
            if self.num_frames > 0:
                if self.gui_playing.value:
                    self.gui_playing.value = False
                next_step = (int(self.gui_timestep.value) + 1) % self.num_frames
                self.gui_timestep.value = next_step
                self._update_current_frame_image(next_step)
                self.update_frame_visibility()
                self.server.flush()

        @self.gui_prev_frame.on_click
        def _(_) -> None:
            if self._saving_video:
                return
            if self.num_frames > 0:
                if self.gui_playing.value:
                    self.gui_playing.value = False
                prev_step = (int(self.gui_timestep.value) - 1) % self.num_frames
                self.gui_timestep.value = prev_step
                self._update_current_frame_image(prev_step)
                self.update_frame_visibility()
                self.server.flush()

        @self.gui_playing.on_update
        def _(_) -> None:
            if self._saving_video:
                return
            self.update_frame_visibility()

        @self.gui_framerate_options.on_click
        def _(_) -> None:
            self.gui_framerate.value = int(self.gui_framerate_options.value)

        @self.gui_timestep.on_update
        def _(event: viser.GuiEvent) -> None:
            if self._saving_video and not self._updating_timestep_from_loop:
                if self._export_timestep is not None:
                    self._updating_timestep_from_loop = True
                    try:
                        self.gui_timestep.value = self._export_timestep
                    finally:
                        self._updating_timestep_from_loop = False
                    self.update_frame_visibility(self._export_timestep)
                    self.server.flush()
                return
            if (
                event.client is not None
                and not self._updating_timestep_from_loop
                and self.gui_playing.value
            ):
                self.gui_playing.value = False
            step_idx = int(self.gui_timestep.value)
            self._update_current_frame_image(step_idx)
            self.update_frame_visibility()

    def _update_current_frame_image(self, step_idx: int) -> None:
        if (
            self.current_frame_image is None
            or not self.show_video_checkbox.value
            or step_idx == self._last_displayed_image_idx
        ):
            return
        if not (0 <= step_idx < len(self.original_images)):
            return
        self._last_displayed_image_idx = step_idx
        self.current_frame_image.image = self.original_images[step_idx]

    def _build_scene(self) -> None:
        """Create `/frames/{step}` nodes with filtered point clouds and cameras."""
        with self._lock, self.server.atomic():
            self.server.scene.add_frame("/frames", show_axes=False)
            cam_stride = int(self.camera_downsample_slider.value)
            for i, step in enumerate(self.all_steps):
                self.frame_nodes.append(
                    self.server.scene.add_frame(f"/frames/{step}", show_axes=False)
                )
                self.add_pc(step)
                if self.show_camera and i % cam_stride == 0:
                    self.add_camera(step)
            self.update_frame_visibility()

    def _report_display_error(self, action: str, error: Exception) -> None:
        """Log and surface background display-update failures in the UI."""
        logger.exception("Viser %s failed", action)
        message = f"Error ({action}): {error}"
        self.last_display_error = message
        self.display_status.value = message
        self.display_status.visible = True
        for client in list(self.server.get_clients().values()):
            if hasattr(client, "add_notification"):
                try:
                    client.add_notification(
                        title=f"{action} failed",
                        body=str(error),
                        color="red",
                        auto_close_seconds=8.0,
                    )
                except Exception:
                    pass
        self.server.flush()

    def _clear_display_error(self, _action: str = "") -> None:
        if self.last_display_error is None:
            return
        if any(task.last_error is not None for task in self._debounced_tasks):
            return
        self.last_display_error = None
        self.display_status.value = "Ready"
        self.display_status.visible = False
        self.server.flush()

    def _apply_focal_length(self, cancel_check=None) -> None:
        if cancel_check is not None and cancel_check():
            return
        with self._lock:
            if cancel_check is not None and cancel_check():
                return
            vfov = self._focal_to_vfov(self.focal_slider.value)
            for client in self.server.get_clients().values():
                client.camera.fov = vfov
                self._sync_client_telemetry(client, force=True)
        self.server.flush()

    def _apply_point_size(self, flush: bool = True, cancel_check=None) -> None:
        if cancel_check is not None and cancel_check():
            return
        with self._lock, self.server.atomic():
            if cancel_check is not None and cancel_check():
                return
            psize = float(self.psize_slider.value)
            for handle in self.pc_handles:
                handle.point_size = psize
        if flush:
            self.server.flush()

    def _apply_camera_size(self, flush: bool = True, cancel_check=None) -> None:
        if cancel_check is not None and cancel_check():
            return
        with self._lock, self.server.atomic():
            if cancel_check is not None and cancel_check():
                return
            scale = float(self.camsize_slider.value)
            thickness = 0.03 * scale
            ratio = scale / 0.03
            for handle in self.cam_handles:
                handle.scale = scale
                _set_frustum_thickness(handle, thickness)
            for axes in self.cam_axes_handles:
                axes.axes_length = 0.05 * ratio
                axes.axes_radius = 0.002 * ratio
                axes.origin_radius = 0.002 * ratio
        if flush:
            self.server.flush()

    def _normalize_pc_inputs(
        self,
        pc,
        color,
        conf=None,
        edge_color=None,
        set_border_color: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Normalize point/color/conf arrays and compute an optional finite-point mask."""
        pred_pts = np.asarray(pc).reshape(-1, 3)
        if set_border_color and edge_color is not None:
            arr = np.asarray(color)
            img = arr[0] if arr.ndim == 4 else arr
            color = self.set_color_border(img, color=edge_color)
        color = np.asarray(color)
        if np.issubdtype(color.dtype, np.floating) and np.isnan(color).any():
            color = np.zeros((pred_pts.shape[0], 3), dtype=np.float32)
            color[:, 2] = 1.0
        else:
            color = color.reshape(-1, 3)

        conf_flat = np.asarray(conf).reshape(-1) if conf is not None else None
        valid = np.isfinite(pred_pts).all(axis=1)
        valid_mask = None if valid.all() else valid
        return pred_pts, color, conf_flat, valid_mask

    @staticmethod
    def _apply_pc_filter(
        pred_pts: np.ndarray,
        color: np.ndarray,
        conf_flat: Optional[np.ndarray],
        valid_mask: Optional[np.ndarray],
        vis_threshold: float,
        downsample_factor: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply finite/confidence masks and return compact downsampled arrays."""
        mask = valid_mask
        if conf_flat is not None:
            conf_mask = conf_flat > vis_threshold
            mask = conf_mask if mask is None else (mask & conf_mask)
        if mask is not None:
            pred_pts = pred_pts[mask]
            color = color[mask]

        if len(pred_pts) == 0:
            return pred_pts, color

        stride = int(downsample_factor)
        if stride > 1:
            # Copy strided slice so the compact array does not retain the full filtered buffer.
            pred_pts = np.ascontiguousarray(pred_pts[::stride])
            color = np.ascontiguousarray(color[::stride])
        return pred_pts, color

    def _filter_step_pc(
        self,
        step: int,
        vis_threshold: float,
        downsample_factor: int,
        set_border_color: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if set_border_color and step in self._pc_base_cache:
            base = self._pc_base_cache[step]
        else:
            entry = self.pcs[step]
            base = self._normalize_pc_inputs(
                entry["pc"],
                entry["color"],
                entry["conf"],
                entry.get("edge_color"),
                set_border_color=set_border_color,
            )
            if set_border_color:
                self._pc_base_cache[step] = base
        pred_pts, color, conf_flat, valid_mask = base
        return self._apply_pc_filter(
            pred_pts,
            color,
            conf_flat,
            valid_mask,
            vis_threshold=vis_threshold,
            downsample_factor=downsample_factor,
        )

    def _regenerate_point_clouds(self, cancel_check=None) -> None:
        """Remove old point clouds and rebuild them with current threshold/downsampling."""
        if not self.frame_nodes:
            return
        threshold = float(self.vis_threshold)
        stride = int(self.downsample_slider.value)

        # Compute filtered arrays outside atomic()/lock so rapid drags can cancel
        # early without stalling the Viser server or sending stale payloads.
        prepared: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for idx, step in enumerate(self.all_steps):
            if cancel_check is not None and idx % 16 == 0 and cancel_check():
                return
            pts, cols = self._filter_step_pc(
                step,
                vis_threshold=threshold,
                downsample_factor=stride,
                set_border_color=True,
            )
            prepared.append((step, pts, cols))

        if cancel_check is not None and cancel_check():
            return

        self._rebuilding = True
        try:
            with self._lock, self.server.atomic():
                if not self.frame_nodes:
                    return
                if cancel_check is not None and cancel_check():
                    return
                # Read latest point size inside the lock so concurrent slider changes are not clobbered.
                psize = float(self.psize_slider.value)
                for handle in self.pc_handles:
                    try:
                        handle.remove()
                    except (KeyError, AttributeError):
                        pass
                self.pc_handles.clear()

                for step, pred_pts, color in prepared:
                    self.pc_handles.append(
                        self.server.scene.add_point_cloud(
                            name=f"/frames/{step}/pred_pts",
                            points=pred_pts,
                            colors=color,
                            point_size=psize,
                        )
                    )
            self.server.flush()
        finally:
            self._rebuilding = False

    def _regenerate_cameras(self, cancel_check=None) -> None:
        """Remove old cameras and rebuild them with current visibility/downsampling."""
        if cancel_check is not None and cancel_check():
            return
        self._rebuilding = True
        try:
            with self._lock, self.server.atomic():
                if not self.frame_nodes or (cancel_check is not None and cancel_check()):
                    return
                for handle in self.cam_handles:
                    try:
                        handle.remove()
                    except (KeyError, AttributeError):
                        pass
                for axes in self.cam_axes_handles:
                    try:
                        axes.remove()
                    except (KeyError, AttributeError):
                        pass
                self.cam_handles.clear()
                self.cam_axes_handles.clear()

                if self.show_camera:
                    downsample_factor = int(self.camera_downsample_slider.value)
                    for i, step in enumerate(self.all_steps):
                        if i % downsample_factor == 0:
                            self.add_camera(step)
            self.server.flush()
        finally:
            self._rebuilding = False

    def _focal_to_vfov(self, focal: float) -> float:
        """Convert slider focal length (pixels for sensor height `self.size`) to vertical FOV (rad)."""
        f = max(float(focal), 1e-3)
        vfov = float(2.0 * np.arctan((0.5 * float(self.size)) / f))
        return float(np.clip(vfov, np.deg2rad(1.0), np.deg2rad(175.0)))

    def _camera_vfov_and_aspect(self, step: int) -> Tuple[float, float]:
        """Compute vertical FOV (rad) and aspect ratio (W/H) for a reconstructed camera."""
        cam = self.cam_dict
        fx = float(cam["focal"][step]) if cam and "focal" in cam else 1.0
        fy = float(cam["fy"][step]) if cam and "fy" in cam else fx
        pp = cam["pp"][step] if cam and "pp" in cam else (1.0, 1.0)
        if cam and "image_size" in cam:
            img_h, img_w = cam["image_size"][step]
            if img_h > 0 and img_w > 0:
                vfov = float(2.0 * np.arctan((0.5 * float(img_h)) / max(fy, 1e-6)))
                aspect = float((float(img_w) * fy) / max(float(img_h) * fx, 1e-6))
                return vfov, aspect
        vfov = float(2.0 * np.arctan(float(pp[1]) / max(fy, 1e-6)))
        aspect = float((float(pp[0]) * fy) / max(float(pp[1]) * fx, 1e-6))
        return vfov, aspect

    @staticmethod
    def _safe_camera_fov(camera: Any, fallback: float) -> float:
        try:
            fov = float(camera.fov)
            if np.isfinite(fov) and fov > 0:
                return fov
        except Exception:
            pass
        return fallback

    @staticmethod
    def _safe_camera_aspect(camera: Any, fallback: float = 1.0) -> float:
        try:
            aspect = float(camera.aspect)
            if np.isfinite(aspect) and aspect > 0:
                return aspect
        except Exception:
            pass
        return fallback

    def _sync_client_telemetry(
        self, client: viser.ClientHandle, force: bool = False
    ) -> None:
        client_key = getattr(client, "client_id", id(client))
        panels = self._client_telemetry.get(client_key)
        if not panels:
            return
        now = time.monotonic()
        if not force and (now - self._telemetry_last_sync.get(client_key, 0.0)) < 0.2:
            return
        self._telemetry_last_sync[client_key] = now
        try:
            wxyz_str = f"{client.camera.wxyz}"
            pos_str = f"{client.camera.position}"
        except Exception:
            wxyz_str, pos_str = "", ""
        fov_rad = self._safe_camera_fov(
            client.camera, self._focal_to_vfov(self.focal_slider.value)
        )
        aspect = self._safe_camera_aspect(client.camera, 1.0)
        with self.server.atomic():
            if wxyz_str:
                panels["wxyz"].value = wxyz_str
            if pos_str:
                panels["position"].value = pos_str
            panels["fov"].value = f"{np.degrees(fov_rad):.2f}"
            panels["aspect"].value = f"{aspect:.3f}"

    def update_frame_visibility(self, timestep: Optional[int] = None) -> None:
        """Show frames <= timestep (3D) or == timestep (4D), sending minimal diffs."""
        with self._lock:
            if not self.frame_nodes:
                return
            if timestep is not None:
                raw_step = int(timestep)
            elif self._saving_video and self._export_timestep is not None:
                raw_step = int(self._export_timestep)
            else:
                raw_step = int(self.gui_timestep.value)
            current_timestep = min(max(raw_step, 0), len(self.frame_nodes) - 1)
            with self.server.atomic():
                for i, frame_node in enumerate(self.frame_nodes):
                    visible = (
                        i == current_timestep if self.fourd else i <= current_timestep
                    )
                    if frame_node.visible != visible:
                        frame_node.visible = visible

    def _move_to_camera(self, frame_idx: int, smooth: bool = True) -> None:
        """Move viewer camera to match reconstructed camera at given frame."""
        if self.cam_dict is None or not self.all_steps:
            return
        idx = min(max(frame_idx, 0), len(self.all_steps) - 1)
        step = self.all_steps[idx]
        rot = self.cam_dict["R"][step] if "R" in self.cam_dict else np.eye(3)
        trans = self.cam_dict["t"][step] if "t" in self.cam_dict else np.zeros(3)

        offset = 0.5
        viewing_dir = rot[:, 2]
        position = trans - viewing_dir * offset
        look_at = trans + viewing_dir * 0.5
        fov, _ = self._camera_vfov_and_aspect(step)
        up = -rot[:, 1]

        for client in self.server.get_clients().values():
            if smooth:
                self._smooth_camera_transition(
                    client,
                    target_position=position,
                    target_look_at=look_at,
                    target_up=up,
                    target_fov=fov,
                    duration=0.3,
                )
            else:
                client.camera.up_direction = tuple(up)
                client.camera.position = tuple(position)
                client.camera.look_at = tuple(look_at)
                client.camera.fov = fov
                self._sync_client_telemetry(client, force=True)

    def _smooth_camera_transition(
        self,
        client: viser.ClientHandle,
        target_position,
        target_look_at=None,
        target_up=None,
        target_fov=None,
        duration: float = 0.3,
    ) -> None:
        """Smoothly ease camera position, look-at point and optional FOV."""

        def interpolate():
            try:
                num_steps = 15
                dt = duration / num_steps
                start_position = np.array(client.camera.position, dtype=np.float64)
                start_look_at = np.array(client.camera.look_at, dtype=np.float64)
                start_fov = self._safe_camera_fov(
                    client.camera,
                    target_fov
                    if target_fov is not None
                    else self._focal_to_vfov(self.focal_slider.value),
                )

                end_position = np.asarray(target_position, dtype=np.float64)
                end_look_at = (
                    np.asarray(target_look_at, dtype=np.float64)
                    if target_look_at is not None
                    else start_look_at
                )

                if target_up is not None:
                    client.camera.up_direction = tuple(
                        np.asarray(target_up, dtype=np.float64)
                    )

                for i in range(num_steps + 1):
                    alpha = i / num_steps
                    alpha_smooth = alpha * alpha * (3 - 2 * alpha)
                    interp_pos = (
                        start_position + (end_position - start_position) * alpha_smooth
                    )
                    interp_look = (
                        start_look_at + (end_look_at - start_look_at) * alpha_smooth
                    )
                    client.camera.position = tuple(interp_pos)
                    client.camera.look_at = tuple(interp_look)
                    if target_fov is not None:
                        client.camera.fov = (
                            start_fov + (target_fov - start_fov) * alpha_smooth
                        )
                    time.sleep(dt)
                self._sync_client_telemetry(client)
            except Exception:
                return

        Thread(target=interpolate, daemon=True).start()

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        """Get current camera state from a connected Viser client."""
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [tf.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], axis=1
                ),
                [[0.0, 0.0, 0.0, 1.0]],
            ],
            axis=0,
        )
        return CameraState(fov=camera.fov, aspect=camera.aspect, c2w=c2w)

    @staticmethod
    def generate_pseudo_intrinsics(h: int, w: int) -> np.ndarray:
        """Generate pseudo intrinsics from image size."""
        focal = (h**2 + w**2) ** 0.5
        return np.array(
            [[focal, 0, w // 2], [0, focal, h // 2], [0, 0, 1]], dtype=np.float32
        )

    def _connect_client(self, client: viser.ClientHandle) -> None:
        """Setup initial view and per-client camera telemetry panel."""
        self._set_view(client)

        init_fov = self._safe_camera_fov(
            client.camera, self._focal_to_vfov(self.focal_slider.value)
        )
        init_aspect = self._safe_camera_aspect(client.camera, 1.0)
        wxyz_panel = client.gui.add_text("wxyz:", f"{client.camera.wxyz}")
        position_panel = client.gui.add_text("position:", f"{client.camera.position}")
        fov_panel = client.gui.add_text("fov:", f"{np.degrees(init_fov):.2f}")
        aspect_panel = client.gui.add_text("aspect:", f"{init_aspect:.3f}")
        client_key = getattr(client, "client_id", id(client))
        self._client_telemetry[client_key] = {
            "wxyz": wxyz_panel,
            "position": position_panel,
            "fov": fov_panel,
            "aspect": aspect_panel,
        }

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            self._sync_client_telemetry(client)

    def _disconnect_client(self, client: viser.ClientHandle) -> None:
        client_key = getattr(client, "client_id", id(client))
        self._client_telemetry.pop(client_key, None)
        self._telemetry_last_sync.pop(client_key, None)

    @staticmethod
    def set_color_border(image, border_width=5, color=(1, 0, 0)):
        """Add colored border to an image array."""
        image = np.array(image, copy=True)
        for ch in range(3):
            image[:border_width, :, ch] = color[ch]
            image[-border_width:, :, ch] = color[ch]
            image[:, :border_width, ch] = color[ch]
            image[:, -border_width:, ch] = color[ch]
        return image

    def read_data(self, pc_list, color_list, conf_list, edge_color_list=None):
        """Organize per-frame point cloud data and viridis camera colors."""
        pcs = {}
        step_list = []
        for i, pc in enumerate(pc_list):
            pcs[i] = {
                "pc": pc,
                "color": color_list[i],
                "conf": conf_list[i] if conf_list is not None else None,
                "edge_color": (
                    None
                    if edge_color_list is None or edge_color_list[i] is None
                    else edge_color_list[i]
                ),
            }
            step_list.append(i)

        num_cameras = len(pc_list)
        if num_cameras > 1:
            normalized = np.arange(num_cameras) / (num_cameras - 1)
        else:
            normalized = np.array([0.0])
        self.camera_colors = colormaps["viridis"](normalized)
        return pcs, step_list

    def parse_pc_data(
        self,
        pc,
        color,
        conf=None,
        edge_color=(0.251, 0.702, 0.902),
        set_border_color: bool = False,
        downsample_factor: int = 1,
    ):
        """Filter finite points by absolute confidence threshold, then subsample."""
        pred_pts, color_arr, conf_flat, valid_mask = self._normalize_pc_inputs(
            pc,
            color,
            conf,
            edge_color=edge_color,
            set_border_color=set_border_color,
        )
        return self._apply_pc_filter(
            pred_pts,
            color_arr,
            conf_flat,
            valid_mask,
            vis_threshold=float(self.vis_threshold),
            downsample_factor=int(downsample_factor),
        )

    def add_pc(self, step: int) -> None:
        """Add filtered point cloud under `/frames/{step}/pred_pts`."""
        pred_pts, color = self._filter_step_pc(
            step,
            vis_threshold=float(self.vis_threshold),
            downsample_factor=int(self.downsample_slider.value),
            set_border_color=True,
        )
        self.pc_handles.append(
            self.server.scene.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=color,
                point_size=self.psize_slider.value,
            )
        )

    def add_camera(self, step: int) -> None:
        """Add camera axes and untextured viridis frustum under `/frames/{step}`."""
        cam = self.cam_dict
        rot = cam["R"][step] if cam and "R" in cam else np.eye(3)
        trans = cam["t"][step] if cam and "t" in cam else np.zeros(3)

        q = tf.SO3.from_matrix(rot).wxyz
        fov, aspect = self._camera_vfov_and_aspect(step)

        step_index = self.all_steps.index(step) if step in self.all_steps else 0
        camera_color = self.camera_colors[step_index]
        camera_color_rgb = tuple(int(c * 255) for c in camera_color[:3])

        scale = self.camsize_slider.value
        ratio = scale / 0.03
        axes_handle = self.server.scene.add_frame(
            f"/frames/{step}/camera_frame",
            wxyz=q,
            position=trans,
            axes_length=0.05 * ratio,
            axes_radius=0.002 * ratio,
            origin_radius=0.002 * ratio,
            visible=self.show_camera,
        )
        frustum_handle = self.server.scene.add_camera_frustum(
            name=f"/frames/{step}/camera",
            fov=fov,
            aspect=aspect,
            wxyz=q,
            position=trans,
            scale=scale,
            color=camera_color_rgb,
            visible=self.show_camera,
        )
        _set_frustum_thickness(frustum_handle, 0.03 * scale)

        @frustum_handle.on_click
        def _(event) -> None:
            look_at_pt = trans + rot[:, 2] * 0.5
            up_dir = -rot[:, 1]
            targets = (
                [event.client]
                if getattr(event, "client", None) is not None
                else list(self.server.get_clients().values())
            )
            for client in targets:
                client.camera.up_direction = tuple(up_dir)
                client.camera.position = tuple(trans)
                client.camera.look_at = tuple(look_at_pt)
                client.camera.fov = fov
                self._sync_client_telemetry(client, force=True)

        self.cam_axes_handles.append(axes_handle)
        self.cam_handles.append(frustum_handle)

    def _export_glb(self) -> None:
        """Export current filtered point clouds and cameras as a GLB file."""
        try:
            import trimesh
        except ImportError:
            self.glb_status.value = "Error: pip install trimesh"
            self.server.flush()
            return

        if self.glb_export_button.disabled:
            return
        self.glb_export_button.disabled = True
        self.glb_status.value = "Collecting points..."
        self.server.flush()
        try:
            all_points, all_colors = [], []
            downsample = int(self.downsample_slider.value)
            for step in self.all_steps:
                entry = self.pcs[step]
                pts, cols = self.parse_pc_data(
                    entry["pc"],
                    entry["color"],
                    entry["conf"],
                    entry.get("edge_color"),
                    set_border_color=False,
                    downsample_factor=downsample,
                )
                if len(pts) > 0:
                    all_points.append(pts)
                    if cols.dtype != np.uint8:
                        cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
                    all_colors.append(cols)

            if not all_points:
                self.glb_status.value = "Error: no points to export"
                self.server.flush()
                return

            vertices = np.concatenate(all_points, axis=0)
            colors_rgb = np.concatenate(all_colors, axis=0)

            colors_float = colors_rgb.astype(np.float32) / 255.0
            sat_boost = self.glb_saturation_slider.value
            if sat_boost != 1.0:
                gray = colors_float.mean(axis=1, keepdims=True)
                colors_float = gray + sat_boost * (colors_float - gray)

            bri_boost = self.glb_brightness_slider.value
            if bri_boost != 1.0:
                colors_float = colors_float * bri_boost

            colors_float = np.clip(colors_float, 0.0, 1.0)

            alpha = float(np.clip(self.glb_opacity_slider.value, 0.0, 1.0))
            colors_u8 = np.rint(colors_float * 255.0).clip(0, 255).astype(np.uint8)
            alpha_u8 = int(round(alpha * 255.0))
            colors_rgba = np.concatenate(
                [
                    colors_u8,
                    np.full((len(colors_u8), 1), alpha_u8, dtype=np.uint8),
                ],
                axis=1,
            )

            lo = np.percentile(vertices, 5, axis=0)
            hi = np.percentile(vertices, 95, axis=0)
            scene_scale = max(float(np.linalg.norm(hi - lo)), 0.1)

            scene_3d = trimesh.Scene()
            export_mode = self.glb_mode_dropdown.value
            if export_mode == "Spheres":
                self.glb_status.value = "Building spheres..."
                self.server.flush()
                max_pts = int(self.glb_max_sphere_pts_slider.value)
                radius = float(self.glb_sphere_radius_slider.value)
                if len(vertices) > max_pts:
                    rng = np.random.default_rng(0)
                    idx = np.sort(rng.choice(len(vertices), max_pts, replace=False))
                    vertices = vertices[idx]
                    colors_rgba = colors_rgba[idx]

                sphere_template = trimesh.creation.icosphere(
                    subdivisions=1, radius=radius
                )
                n_verts_per = len(sphere_template.vertices)
                n_faces_per = len(sphere_template.faces)
                all_verts = (
                    vertices[:, None, :].astype(np.float32)
                    + sphere_template.vertices[None, :, :].astype(np.float32)
                ).reshape(-1, 3)
                offsets = (np.arange(len(vertices), dtype=np.int64) * n_verts_per)[
                    :, None, None
                ]
                all_faces = (sphere_template.faces[None, :, :] + offsets).reshape(-1, 3)
                all_face_colors = np.repeat(colors_rgba, n_faces_per, axis=0)

                mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
                mesh.visual.face_colors = all_face_colors
                if alpha < 1.0 and hasattr(mesh.visual, "material"):
                    mesh.visual.material.alphaMode = "BLEND"
                scene_3d.add_geometry(mesh, geom_name="pointcloud")
            else:
                scene_3d.add_geometry(
                    trimesh.PointCloud(vertices=vertices, colors=colors_rgba),
                    geom_name="pointcloud",
                )

            if self.glb_show_cam_checkbox.value and self.cam_dict is not None:
                colormap = colormaps["gist_rainbow"]
                num_cameras = len(self.all_steps)
                cam_positions = []
                frustum_thickness = float(self.glb_frustum_thickness_slider.value)
                effective_cam_scale = scene_scale * float(
                    self.glb_cam_scale_slider.value
                )

                for i, step in enumerate(self.all_steps):
                    rot = self.cam_dict["R"][step] if "R" in self.cam_dict else np.eye(3)
                    trans = self.cam_dict["t"][step] if "t" in self.cam_dict else np.zeros(3)
                    c2w = np.eye(4)
                    c2w[:3, :3] = rot
                    c2w[:3, 3] = trans
                    cam_positions.append(np.asarray(trans, dtype=np.float64))
                    rgba_c = colormap(i / max(num_cameras - 1, 1))
                    cam_color = tuple(int(255 * x) for x in rgba_c[:3])
                    integrate_camera_into_scene(
                        scene_3d,
                        c2w,
                        cam_color,
                        effective_cam_scale,
                        frustum_thickness=frustum_thickness,
                    )

                if self.glb_trajectory_checkbox.value and len(cam_positions) >= 2:
                    traj_radius = float(self.glb_trajectory_radius_slider.value) * float(
                        self.glb_cam_scale_slider.value
                    )
                    traj_mesh = build_trajectory_tube(
                        np.asarray(cam_positions), traj_radius, colormap, num_cameras
                    )
                    if traj_mesh is not None:
                        scene_3d.add_geometry(traj_mesh)

            if self.cam_dict is not None and self.all_steps:
                step0 = self.all_steps[0]
                rot0 = self.cam_dict["R"][step0] if "R" in self.cam_dict else np.eye(3)
                trans0 = self.cam_dict["t"][step0] if "t" in self.cam_dict else np.zeros(3)
                c2w_0 = np.eye(4)
                c2w_0[:3, :3] = rot0
                c2w_0[:3, 3] = trans0
                extrinsics = np.expand_dims(np.linalg.inv(c2w_0), 0)
                scene_3d = apply_scene_alignment(scene_3d, extrinsics)

            raw_output = self.glb_output_path.value.strip()
            if not raw_output:
                self.glb_status.value = "Error: empty output path"
                self.server.flush()
                return
            output_path = Path(raw_output).expanduser()
            if output_path.parent and str(output_path.parent) not in ("", "."):
                output_path.parent.mkdir(parents=True, exist_ok=True)
            if alpha < 1.0 and output_path.suffix.lower() in ("", ".glb"):
                glb_bytes = scene_3d.export(file_type="glb")
                glb_bytes = _set_glb_mesh_alpha_mode(
                    glb_bytes, mesh_name="pointcloud", alpha_mode="BLEND"
                )
                output_path.write_bytes(glb_bytes)
            else:
                scene_3d.export(str(output_path))
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError(f"failed to write {output_path}")

            n_pts = len(vertices)
            mode_str = (
                f"spheres r={self.glb_sphere_radius_slider.value}"
                if export_mode == "Spheres"
                else "points"
            )
            self.glb_status.value = f"Saved: {output_path} ({n_pts:,} {mode_str})"
            self.server.flush()
        except Exception as error:
            self.glb_status.value = f"Error: {error}"
            self.server.flush()
        finally:
            self.glb_export_button.disabled = False
            self.server.flush()

    @staticmethod
    def _capture_render(client, width: int, height: int, timeout: float = 30.0):
        try:
            return client.camera.get_render(
                height=height, width=width, timeout=timeout
            )
        except TypeError:
            return client.camera.get_render(height=height, width=width)

    def _capture_video_frame(
        self,
        client: Any,
        width: int,
        height: int,
        frame_idx: int,
        total_frames: int,
        timeout: float = 45.0,
    ):
        """Capture one video frame with a single retry on transient TimeoutError."""
        try:
            return self._capture_render(client, width, height, timeout=timeout)
        except TimeoutError:
            self.video_status.value = (
                f"Waiting for browser at {frame_idx + 1}/{total_frames} "
                "(keep tab visible)..."
            )
            self.server.flush()
            try:
                return self._capture_render(client, width, height, timeout=timeout)
            except TimeoutError as error:
                raise TimeoutError(
                    f"{error} (keep browser tab active/visible during export)"
                ) from error

    @staticmethod
    def _resolve_screenshot_resolution(res_str: str, client: Any) -> Tuple[int, int]:
        """Resolve screenshot (width, height), using real client viewport for 'Current'."""
        if res_str != "Current":
            width, height = map(int, res_str.split("x"))
            return width, height
        camera = getattr(client, "camera", None)
        if camera is not None:
            try:
                raw_w = getattr(camera, "image_width", None)
                raw_h = getattr(camera, "image_height", None)
                if (
                    isinstance(raw_w, (int, np.integer))
                    and not isinstance(raw_w, bool)
                    and isinstance(raw_h, (int, np.integer))
                    and not isinstance(raw_h, bool)
                    and int(raw_w) > 0
                    and int(raw_h) > 0
                ):
                    return int(raw_w), int(raw_h)
            except Exception:
                pass
            try:
                aspect = getattr(camera, "aspect", None)
                if (
                    isinstance(aspect, (int, float, np.floating, np.integer))
                    and not isinstance(aspect, bool)
                    and np.isfinite(float(aspect))
                    and float(aspect) > 0
                ):
                    return max(1, int(round(1080.0 * float(aspect)))), 1080
            except Exception:
                pass
        return 1920, 1080

    def _take_screenshot(self, client: Optional[Any] = None) -> None:
        """Capture a screenshot from the current browser view and save to disk."""
        raw_path = self.screenshot_path.value.strip()
        if not raw_path:
            self.screenshot_status.value = "Error: empty save path"
            self.server.flush()
            return
        output_path = Path(raw_path).expanduser()
        res_str = self.screenshot_resolution.value

        if client is None:
            clients = list(self.server.get_clients().values())
            if not clients:
                self.screenshot_status.value = "Error: no client connected"
                self.server.flush()
                return
            client = clients[0]

        self.screenshot_button.disabled = True
        try:
            self.screenshot_status.value = "Capturing..."
            self.server.flush()
            width, height = self._resolve_screenshot_resolution(res_str, client)

            render = self._capture_render(client, width, height, timeout=30.0)
            if render is None:
                self.screenshot_status.value = "Error: render returned None"
                self.server.flush()
                return

            frame = np.asarray(render)
            if frame.ndim != 3 or frame.shape[2] not in (3, 4) or frame.size == 0:
                self.screenshot_status.value = "Error: invalid render frame"
                self.server.flush()
                return
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if output_path.parent and str(output_path.parent) not in ("", "."):
                output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), frame_bgr) or not output_path.is_file():
                self.screenshot_status.value = f"Error: failed to write {output_path}"
                self.server.flush()
                return
            self.screenshot_status.value = f"Saved: {output_path} ({width}x{height})"
            self.server.flush()
        except Exception as error:
            self.screenshot_status.value = f"Error: {error}"
            self.server.flush()
        finally:
            self.screenshot_button.disabled = False
            self.server.flush()

    def save_video(
        self,
        output_path: str = "output_pointcloud.mp4",
        fps: int = 30,
        resolution: str = "1920x1080",
        save_original_video: bool = True,
        client: Optional[Any] = None,
    ) -> None:
        """Render timeline frames 0..N-1 and encode an MP4 video with FFmpeg."""
        if self._saving_video:
            return
        self._saving_video = True
        locked_controls = (
            self.save_video_button,
            self.btn_reset_defaults,
            self.gui_timestep,
            self.gui_next_frame,
            self.gui_prev_frame,
            self.gui_playing,
            self.btn_4d,
            self.btn_3d,
        )
        prev_disabled = [ctrl.disabled for ctrl in locked_controls]
        for ctrl in locked_controls:
            ctrl.disabled = True
        temp_dir = None
        previous_replay = self.on_replay
        self.on_replay = True
        try:
            self.video_status.value = "Saving video (keep tab visible)..."
            self.server.flush()
            if not output_path or not str(output_path).strip():
                self.video_status.value = "Error: empty output path"
                self.server.flush()
                return
            ffmpeg_bin = _find_ffmpeg()
            if ffmpeg_bin is None:
                self.video_status.value = "Error: ffmpeg not found"
                self.server.flush()
                return

            if self.num_frames == 0:
                self.video_status.value = "Error: no frames to render"
                self.server.flush()
                return

            width, height = map(int, resolution.split("x"))
            if client is None:
                deadline = time.monotonic() + 10.0
                while len(self.server.get_clients()) == 0:
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "No client connected. Please open the visualization in a browser first."
                        )
                    time.sleep(0.1)
                client = next(iter(self.server.get_clients().values()))

            temp_dir = tempfile.mkdtemp(prefix="viser_video_")
            for i in range(self.num_frames):
                if i == 0 or (i + 1) % 5 == 0 or i + 1 == self.num_frames:
                    self.video_status.value = (
                        f"Rendering {i + 1}/{self.num_frames} (keep tab visible)..."
                    )
                with self._lock:
                    self._export_timestep = i
                    self._updating_timestep_from_loop = True
                    try:
                        self.gui_timestep.value = i
                    finally:
                        self._updating_timestep_from_loop = False
                    self.update_frame_visibility(i)
                self.server.flush()
                time.sleep(0.02)

                # Re-assert locked timestep right before capture in case an external
                # mutation raced with the sleep window.
                if int(self.gui_timestep.value) != i:
                    with self._lock:
                        self._updating_timestep_from_loop = True
                        try:
                            self.gui_timestep.value = i
                        finally:
                            self._updating_timestep_from_loop = False
                        self.update_frame_visibility(i)
                    self.server.flush()

                try:
                    screenshot = self._capture_video_frame(
                        client,
                        width,
                        height,
                        frame_idx=i,
                        total_frames=self.num_frames,
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"render failed at frame {i + 1}/{self.num_frames}: {error}"
                    ) from error
                if screenshot is None:
                    raise RuntimeError(
                        f"render returned None at frame {i + 1}/{self.num_frames}"
                    )
                frame = np.asarray(screenshot)
                if frame.ndim != 3 or frame.shape[2] not in (3, 4) or frame.size == 0:
                    raise RuntimeError(
                        f"invalid render frame at {i + 1}/{self.num_frames}"
                    )
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                if not cv2.imwrite(frame_path, frame):
                    raise RuntimeError(f"failed to write temporary frame {frame_path}")

            self.video_status.value = "Encoding video..."
            self.server.flush()
            target_path = Path(str(output_path).strip()).expanduser()
            if target_path.parent and str(target_path.parent) not in ("", "."):
                target_path.parent.mkdir(parents=True, exist_ok=True)

            ffmpeg_cmd = [
                ffmpeg_bin,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(target_path),
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if (
                result.returncode != 0
                or not target_path.is_file()
                or target_path.stat().st_size == 0
            ):
                stderr_lines = (result.stderr or "").strip().splitlines()
                detail = stderr_lines[-1] if stderr_lines else f"exit code {result.returncode}"
                self.video_status.value = f"Error: FFmpeg failed ({detail})"
                self.server.flush()
                return

            if save_original_video and self.original_images:
                self._save_original_video(
                    str(target_path), int(fps), width, height, ffmpeg_bin
                )

            self.video_status.value = f"Saved to {target_path}"
            self.server.flush()
        except Exception as error:
            self.video_status.value = f"Error: {error}"
            self.server.flush()
        finally:
            self._export_timestep = None
            self.on_replay = previous_replay
            self._saving_video = False
            for ctrl, was_disabled in zip(locked_controls, prev_disabled):
                ctrl.disabled = was_disabled
            self._update_current_frame_image(int(self.gui_timestep.value))
            self.update_frame_visibility()
            self.server.flush()
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _save_original_video(
        self,
        pointcloud_video_path: str,
        fps: int,
        width: int,
        height: int,
        ffmpeg_bin: str,
    ) -> str:
        """Save original processed RGB frames alongside the point cloud video."""
        base_path = os.path.splitext(pointcloud_video_path)[0]
        original_video_path = f"{base_path}_original.mp4"
        temp_dir = tempfile.mkdtemp(prefix="original_video_")
        try:
            for i, img in enumerate(self.original_images):
                frame = cv2.resize(img, (width, height))
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                if not cv2.imwrite(frame_path, frame):
                    raise RuntimeError(
                        f"failed to write temporary RGB frame {frame_path}"
                    )

            result = subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(temp_dir, "frame_%06d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    original_video_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if (
                result.returncode != 0
                or not os.path.isfile(original_video_path)
                or os.path.getsize(original_video_path) == 0
            ):
                stderr_lines = (result.stderr or "").strip().splitlines()
                detail = (
                    stderr_lines[-1]
                    if stderr_lines
                    else f"exit code {result.returncode}"
                )
                raise RuntimeError(f"FFmpeg failed for original video ({detail})")
            return original_video_path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def animate(self) -> None:
        """Run the V1 playback loop until close() or KeyboardInterrupt."""
        while not self._closed:
            if not self.server.get_clients():
                time.sleep(0.1)
                continue
            if (
                self._rebuilding
                or self._pc_rebuilder.pending
                or self._cam_rebuilder.pending
            ):
                time.sleep(0.05)
                continue
            if not self.on_replay and self.num_frames > 0:
                if self.gui_playing.value:
                    next_step = (int(self.gui_timestep.value) + 1) % self.num_frames
                    self._updating_timestep_from_loop = True
                    try:
                        self.gui_timestep.value = next_step
                    finally:
                        self._updating_timestep_from_loop = False
                    self._update_current_frame_image(next_step)
                    self.update_frame_visibility()
            time.sleep(1.0 / max(float(self.gui_framerate.value), 1.0))

    def run(self, background_mode: bool = False) -> None:
        """Start the interactive playback loop."""
        if background_mode:
            Thread(target=self.animate, daemon=True).start()
        else:
            self.animate()

    def close(self) -> None:
        self._closed = True
        for task in getattr(self, "_debounced_tasks", ()):
            task.stop()
        self.server.stop()
