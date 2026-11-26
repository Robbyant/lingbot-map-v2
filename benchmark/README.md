# LingBot-Map 2.0 Benchmark

**Geometric Streaming Base Model**

Evaluate `GCRStream` on the paper's seven datasets and nine trajectory, scene-pose
and dense-reconstruction settings. The pipeline has three stages:
**prepare** (data and frame selection), **run** (Online inference), and
**evaluate** (metrics and dataset summaries).

This benchmark uses pure causal Online checkpoints. The matching checkpoint and
real-weight GPU results are pending release. Ground truth is used for preparation
and scoring only; it is never an input to the model.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Datasets](#supported-datasets)
3. [Data Preparation](#data-preparation)
4. [Configuration System](#configuration-system)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Interrupt and Resume](#interrupt-and-resume)
7. [Output Layout](#output-layout)
8. [Dense References and Shared ROI](#dense-references-and-shared-roi)
9. [Explicit Manifests (Advanced)](#explicit-manifests-advanced)

## Quick Start

Use the same three-stage workflow as LingBot-Map: configure paths, prepare data,
run the model, then evaluate. Only this paper's datasets and the native Map2
`GCRStream` method are included. One environment is sufficient:

First install a CUDA-enabled PyTorch >= 2.8 build compatible with your GPU and
driver, following the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
Then install the benchmark dependencies from the repository root:

```bash
pip install -e '.[benchmark]'
cd benchmark
```

Alternatively, from the repository root, create the same environment with:

```bash
bash benchmark/envs/install_lingbot_map2.sh
# For an existing lingbot-map2 environment with CUDA PyTorch >= 2.8:
bash benchmark/envs/install_lingbot_map2.sh --append
```

The script installs this package and its benchmark extra in one environment;
it never removes an existing environment. `pyproject.toml` is the dependency
source of truth. CUDA PyTorch installation is documented separately in the
[root README](../README.md#installation), following LingBot-Map's setup style.

### 1. Get and prepare the data

See [Data preparation](#data-preparation) below. There is no unified download of
all benchmark datasets. A model/demo dataset on Hugging Face is not a substitute
for the full RGB, calibration, poses and scanner references required here.

### 2. Configure paths

- `configs/datasets/<dataset>.yaml`: set `raw_data_root`; dense settings also need
  `reference_root`. Optionally select `scenes` for a partial run.
- `configs/methods/lingbot_map2.yaml`: set `_checkpoint` and, if needed, `_config`
  for the matching pure causal checkpoint JSON. `_device` and `_compile` control
  execution. `_depth: false` enables pose-only accuracy runs; dense reconstruction
  requires `_depth: true`.
- `configs/<dataset>.yaml`: lists `datasets`, `methods` and `workspace`. The
  default workspace is the repository's `outputs/benchmark/` directory.

Paths resolve relative to the YAML in which they appear. If moving a base YAML
outside `configs/`, set its optional `config_root` to the directory containing
`datasets/` and `methods/`. The YAML parser rejects unknown fields.

### 3. Launch

```bash
# CPU-only layout/coverage check; no weights or output directory needed.
python prepare.py --config configs/kitti.yaml --check-only

python prepare.py  --config configs/kitti.yaml
python run.py      --config configs/kitti.yaml
python evaluate.py --config configs/kitti.yaml
```

Replace `kitti.yaml` with another base config; the commands stay the same.
Preparation writes the normalized plan automatically. Users do not need to write
per-frame JSON manifests. The stored plan contains ordered source IDs, associated
GT and input hashes; full prediction arrays are saved separately.

| Base config | Setting |
| --- | --- |
| [`configs/kitti.yaml`](configs/kitti.yaml) | KITTI 00–10 trajectory |
| [`configs/vbr.yaml`](configs/vbr.yaml) | Seven VBR trajectories |
| [`configs/oxford.yaml`](configs/oxford.yaml) | Ten full Oxford trajectories |
| [`configs/tat.yaml`](configs/tat.yaml) | Six Tanks and Temples scene-pose sequences |
| [`configs/princeton365.yaml`](configs/princeton365.yaml) | 57 Princeton365 scanning validation sequences |
| [`configs/crocodl.yaml`](configs/crocodl.yaml) | 62 CroCoDL scene-pose sequences |
| [`configs/crocodl_dense.yaml`](configs/crocodl_dense.yaml) | CroCoDL dense reconstruction |
| [`configs/euroc.yaml`](configs/euroc.yaml) | Six EuRoC Vicon-room dense reconstructions |
| [`configs/oxford_dense.yaml`](configs/oxford_dense.yaml) | Oxford metric-frame dense reconstruction |

## Supported Datasets

Sizes below are width × height and use float32 AREA cover-resize and center crop.
The model uses DINOv3, patch16, five special tokens, prefix8, window64 and
FP32 Camera4; the depth head also runs in FP32 when enabled. Dataset-specific settings are in
[datasets/protocols.py](datasets/protocols.py); exact scene lists and raw counts are in
[`configs/`](configs/).

| Protocol | Scenes | Input size | RGB inputs | Report |
| --- | ---: | --- | --- | --- |
| `kitti_trajectory` | 11 | 512×288 | All raw frames | Sim(3) ATE, Micro AUC, RPE |
| `vbr_trajectory` | 7 | 512×288 | All raw frames | Sim(3) ATE, Micro AUC, RPE |
| `oxford_trajectory` | 10 | 512×384 | All raw frames | Sim(3) ATE, Micro AUC, RPE |
| `crocodl_pose` | 62 | 512×384 | `range(0,N,5)` | Micro AUC, ATE |
| `tat_pose` | 6 | 512×288 | All views, 2,185 total | Micro AUC, ATE |
| `princeton365_pose` | 57 | 512×272 | `range(0,N,10)`, 20,370 views | Micro AUC, ATE |
| `crocodl_dense` | 62 | 512×384 | `range(0,N,5)` | CD, F1@0.25m |
| `euroc_dense` | 6 | 512×320 | `range(0,N,5)` | CD, F1@0.25m |
| `oxford_dense` | 10 | 512×384 | Prepared metric RGB, `range(0,N,10)` | CD, F1@4m |

Trajectory inputs and outputs cover every raw frame. The first eight frames are
persistent, followed by indices `8, 8 + ceil(N/320), ...`. Other frames produce
native predictions without changing persistent memory. There is no trajectory
interpolation. The default "Uniform320" rule is an interval rule, not an
exact 320-frame cap. Current plans explicitly record their input scope
(trajectory protocol version 3; pose/dense version 2). Older plans and results
remain in their original workspaces; prepare a new workspace for the new version.
Scene pose and dense protocols retain every selected view;
they do not reset or further subsample long scenes. The runner allocates enough
temporal positions for the whole plan, including a final transient position.
This changes capacity, not token order or existing RoPE phases; quality beyond
the training context length still requires real-weight evaluation.

Oxford trajectory and dense inputs are **different prepared streams**. The
dense split contains about 3,840 source frames per scene (383/384 selected metric
frames); the trajectory split contains 3,961–13,518 raw frames. Do not substitute
one for the other. CroCoDL uses the 62 accepted ARCHE sequences, excluding the two
with inconsistent image-size/orientation metadata. EuRoC uses the six Vicon-room
sequences, not the Machine Hall sequences.

### Keyframe-only trajectory studies

The default `input_scope: full_stream` processes and scores every raw frame.
For an explicitly labeled keyframe-only study, set `input_scope: keyframes` in
`configs/datasets/{kitti,vbr,oxford}.yaml` and choose a new workspace. Preparation
uses the same full source dataset checks, then freezes only Uniform320 frame IDs.
All retained inputs are persistent. An Oxford metric/dense stream is rejected
even in keyframe-only mode; it is not a substitute for the full trajectory source.
Scene-pose and dense protocols keep `selected_views` scope.

Reports state scope, persistence, input size, frame counts, output mode and metric
aggregation. Full-stream and keyframe-only results must not be compared as the
same protocol. Pose-only accuracy runs preserve pose computation but omit DPT
and depth storage; they are not complete-output throughput measurements.

## Data Preparation

Download data from the respective source, then follow the dataset-specific steps
below. Downloading a dataset does not by itself supply the paper's accepted camera
association and common reconstruction ROI.

| Dataset | Get the data | Preparation used here |
| --- | --- | --- |
| Oxford Spires | [Official dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/) · [Hugging Face](https://huggingface.co/datasets/ori-drs/oxford_spires_dataset) | Rectified cam0 RGB and associated C2W; keep full-trajectory and metric-frame trees separate. [Steps](#oxford-spires). |
| VBR | [LoGeR data preparation](https://github.com/Junyi42/LoGeR/blob/main/eval/eval.md#vbr) · [Processed data](https://huggingface.co/datasets/Junyi42/vbr_processed) | Aligned RGB and raw-frame-associated GT. [Steps](#vbr). |
| KITTI | [Official Odometry download](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | Color sequences and pose files, 00–10. [Steps](#kitti). |
| Tanks and Temples | [Official download](https://www.tanksandtemples.org/download/) | Image sets and COLMAP camera logs for the six prescribed scenes. [Steps](#tanks-and-temples). |
| CroCoDL | [Official preparation/download tools](https://github.com/cvg/crocodl-benchmark#4-data) · [Hugging Face](https://huggingface.co/CroCoDL/datasets) | ARCHE iOS sessions, registered camera indexes and NavVis meshes for dense scoring. [Steps](#crocodl). |
| Princeton365 | [Official download instructions](https://github.com/princeton-vl/Princeton365/blob/main/docs/README_download.md) | Scanning validation RGB videos and associated **user-camera** C2W. [Steps](#princeton365). |
| EuRoC MAV | [Official dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/) | Six Vicon-room cam0 sequences, camera/body calibration and Leica references. [Steps](#euroc). |

The `lingbot-map-demo` collection contains demo material, not these complete
benchmark splits. No data is downloaded automatically by the three entrypoints.
Dataset access conditions and licenses remain those of the original providers.

After preparing a dataset, set its `raw_data_root` in `configs/datasets/*.yaml`.
The three entrypoints automatically build the evaluation manifest; no per-image
JSON list is required. Run `prepare.py --config configs/<name>.yaml --check-only`
before preparing or launching inference. It checks the selected scene names,
raw counts, RGB/GT associations and, for dense evaluation, the shared reference.

The adapters consume the layouts below. They do **not** silently infer missing
camera extrinsics, invent timestamp associations, drop invalid-GT RGB frames or
truncate sequences to make the counts match. A different upstream data version
may need a separate preparation or protocol; a count mismatch is an error with
the expected count, not a request to discard frames.

### KITTI

<details>
<summary>Preparation instructions and expected layout</summary>

Get the color odometry sequences and training poses from the
[official Odometry page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
Use sequences 00–10. Point `raw_data_root` to the directory that contains both
`sequences/` and `poses/` (some downloads unpack beneath `dataset/`):

```text
kitti/
  sequences/00/image_2/000000.png
  sequences/00/image_2/000001.png
  ...
  poses/00.txt
  ...
  poses/10.txt
```

No additional image preprocessing is needed. Each pose row is a row-major 3×4
C2W transform; the adapter adds the homogeneous row. It follows the frozen paper
convention of image_2 RGB with the supplied cam0 pose files unchanged. Changing
the stereo-camera convention is a separate protocol, not an implicit correction.

Configure `configs/datasets/kitti.yaml`, then run the three stages with
`configs/kitti.yaml`.

</details>

### VBR

<details>
<summary>Preparation instructions and expected layout</summary>

Follow [LoGeR's VBR preparation](https://github.com/Junyi42/LoGeR/blob/main/eval/eval.md#vbr).
It offers [preprocessed data](https://huggingface.co/datasets/Junyi42/vbr_processed)
and a raw-data preprocessing route. Keep the RGB and GT from the same preparation:

```text
vbr_aligned/
  campus_train0_processed_aligned/rgb/000000.png
  campus_train1_processed_aligned/rgb/000000.png
  ...
  processed_gt/campus_train0_gt.txt
  processed_gt/campus_train1_gt.txt
  ...
```

The seven scene names are listed in `configs/scenes.json`. The GT files use TUM
rows `frame_id tx ty tz qx qy qz qw`. RGB files are sorted by the numeric filename
stem; LoGeR may retain nonzero or nonconsecutive source frame numbers. Files do
not need to be renamed. The GT first column contains **zero-based positions in
this sorted RGB sequence**, before benchmark frame selection, rather than the
source filename numbers. For example, `000003.png`, `000006.png`, `000009.png`
correspond to GT indices `0`, `1`, `2` in the aligned export.
Missing GT indices become NaN poses while all RGB frames remain inputs.
If your download stores seconds or nanoseconds, first use its own association
metadata to map them to RGB indices. The adapter rejects fractional/unassociated
timestamps instead of rounding them.

Configure `configs/datasets/vbr.yaml`; use `configs/vbr.yaml`.

</details>

### Oxford Spires

<details>
<summary>Preparation instructions and expected layout</summary>

Get sequences, calibration and GT from the
[official Oxford Spires project](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/)
or its [Hugging Face dataset](https://huggingface.co/datasets/ori-drs/oxford_spires_dataset).
The [official tools](https://github.com/ori-drs/oxford_spires_dataset) document
acquisition and sensor data handling. For the cam0 rectification/association
workflow, see [LingBot-Map's preprocessing script](https://github.com/Robbyant/lingbot-map/blob/main/preprocess/oxford.py).
That script is a standalone data-preparation reference, not a model or benchmark
dependency of this repository.

For example, in a separate checkout of that preprocessing tool:

```bash
python preprocess/oxford.py \
  --dataset_dir /data/oxford_spires_dataset \
  --output_dir /data/oxford_full \
  --images_only
```

The tool needs the raw `sequences/`, `calibration/` and GT trajectory files. Retain
its timestamp/camera associations and verify coverage against this paper's full
trajectory split before using the output here. Model inference resizing happens
later in `GCRStream`, not in the adapter.

Expected trajectory layout:

```text
oxford_full/
  christ-church-02/
    images/000000.png
    ...
    poses_c2w.txt
    intrinsics.txt
  ...
```

`poses_c2w.txt` has 16 row-major values per image. The full trajectory preset uses
all accepted images; see `configs/frame_counts.json` for the ten required counts.
Select `configs/oxford.yaml`.

**Dense evaluation uses a distinct prepared metric stream.** Point
`configs/datasets/oxford_dense.yaml` to that RGB/pose tree and its `reference_root`.
Do not point it at the full trajectory tree, uniformly resample that tree by eye,
or take its first 3,840 images. The accepted metric stream contains 3,821–3,840
source images per scene and uses every tenth frame. Its original RGB/GT mapping
must be retained. The source dataset's generic projected LiDAR depth is not
interchangeable with the frozen TLS-based reference.

To build a dense reference from already associated metric GT, run from the
repository root:

```bash
python preprocess/dense.py --protocol oxford_dense \
  --root /data/oxford_spires --scene christ-church-02 \
  --output /data/reference_bundles/oxford/christ-church-02 \
  --build-reference --reference-root /data/references/oxford
```

Set `reference_root` in the dataset YAML to `/data/references/oxford`.
The helper reads camera poses, raw-resolution metric depth and intrinsics,
then constructs the 518×388 GT grid and fixed common ROI; see
[Dense references and shared ROI](#dense-references-and-shared-roi).
Raw-data acquisition alone does not provide the associated metric GT depth.

</details>

### Tanks and Temples

<details>
<summary>Preparation instructions and expected layout</summary>

Download the image sets and camera/ground-truth files for **Barn, Caterpillar,
Church, Ignatius, Meetingroom and Truck** from the
[official download page](https://www.tanksandtemples.org/download/). Keep the
provided image subset and COLMAP camera log together; do not sample a new subset
from the high-resolution videos.

```text
tanks_and_temples/
  Barn/
    000001.jpg
    000002.jpg
    ...
    Barn_COLMAP_SfM.log
  Caterpillar/
  Church/
  Ignatius/
  Meetingroom/
  Truck/
```

The adapter reads one metadata line and four C2W matrix rows per log entry.
Image filenames start at one; each image uses log index `int(filename) - 1`.
Numeric gaps are supported: Church contains 507 images with a gap after image
500, while its camera log includes additional entries. Unused camera entries
are ignored; a missing camera for an existing image is an error. No SfM solver
is run and no intrinsics are inferred for the model. The six scenes have 2,185 images in
this paper's split. Configure `configs/datasets/tat.yaml`; use `configs/tat.yaml`.

</details>

### CroCoDL

<details>
<summary>Preparation instructions and expected layout</summary>

Follow the [official full-release preparation/download instructions](https://github.com/cvg/crocodl-benchmark#4-data)
and the [CroCoDL Hugging Face organization](https://huggingface.co/CroCoDL/datasets).
Use the ARCHE_B3, ARCHE_B5, ARCHE_D2 and ARCHE_GRANDE iOS sessions. The challenge
locations and map/query-only sessions are not this paper's split.

The vendor's capture archives require processing/registration before evaluation.
Keep the accepted 62 sessions in `configs/scenes.json`; the two excluded sessions
have inconsistent image-size/orientation metadata. Use the included exporter
from the repository root after the official capture registration step:

```bash
python preprocess/crocodl.py \
  --session /data/capture/ARCHE_B3/sessions/ios_2023-07-12_15.40.46_000 \
  --trajectory /data/capture/ARCHE_B3/registration/ios_2023-07-12_15.40.46_000/trajectory_refined.txt \
  --output /data/crocodl_processed/ARCHE_B3__ios_2023-07-12_15.40.46_000
```

Pass the registered sensor/rig-to-world trajectory explicitly. Its filename can
vary with the upstream registration workflow. The exporter associates exact
timestamps and sensor IDs, composes camera-to-rig extrinsics when needed, and
keeps images without GT. It supports PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL,
RADIAL and OPENCV cameras, producing undistorted PNGs without changing image
orientation. Multicamera sessions require an explicit `--sensor`. It does not
perform scan registration or use phone depth as reconstruction ground truth.

The adapter reads the generated camera index and discovers RGB paths from it:

```text
crocodl_processed/
  ARCHE_B3__ios_2023-07-12_15.40.46_000/
    camera/frames.jsonl
    rgb/...
  ...
```

Each JSONL row identifies one associated raw frame, using the schema in
[Processed camera indexes](#processed-camera-indexes). Export OpenCV C2W from
the registered camera metadata; do not substitute unregistered phone tracking
or phone depth. Sorting is by the exact numeric timestamp of each entire record,
so RGB, intrinsics and pose remain associated.

Use `configs/crocodl.yaml` for scene pose. For `configs/crocodl_dense.yaml`, build
references from the processed cameras and original registered simplified NavVis
meshes. Run from the repository root:

```bash
python preprocess/dense.py --protocol crocodl_dense \
  --root /data/crocodl_processed \
  --scene ARCHE_B3__ios_2023-07-12_15.40.46_000 \
  --geometry /data/meshes/ARCHE_B3/mesh_simplified.ply \
  --output /data/reference_bundles/crocodl/ARCHE_B3__ios_2023-07-12_15.40.46_000 \
  --build-reference --reference-root /data/references/crocodl
```

Repeat for each scene and set the dataset's `reference_root` to
`/data/references/crocodl`. The helper generates camera bundles, common-FOV masks
and flat `<scene>.reference.npz` files.

</details>

### Princeton365

<details>
<summary>Preparation instructions and expected layout</summary>

Use the [official download instructions](https://github.com/princeton-vl/Princeton365/blob/main/docs/README_download.md)
to obtain the validation data. This paper uses 57 object-scanning RGB videos,
listed explicitly in `configs/scenes.json`, not the hidden-GT test set or all
indoor/outdoor categories.

Set `raw_data_root` directly to the official `scanning_videos` directory (or
its parent containing `Benchmark_blur_val/scanning_videos`):

```text
Benchmark_blur_val/scanning_videos/
  user_view/
    new_scanning_user_view_part_1/
      new_scanning_user_view_103.mp4
      relative_gt_to_user_optim.npy
    new_scanning_user_view_part_2/...
    new_scanning_user_view_part_12/...
  gt_view/camera_poses/new_scanning_103.txt
```

No renaming, frame extraction or hand-written GT arrays are required. The
adapter maps the scene name to its reference trajectory and applies the official
user-to-reference calibration by right multiplication:
`user_C2W = reference_C2W @ user_to_reference`. It preserves missing GT as NaNs.
Six videos in this release have one extra final GT row without a corresponding
RGB frame; only that GT-only row is ignored. Every RGB frame remains available.

The extracted official WebDataset layout is also supported:
`[validation/]<sequence>/<sequence>.mp4`, `.gt_trajectory.txt`, and
`.relative_transform.npy`. Scanning sequence names without `_user_view_` are
mapped to the corresponding configured scene names. The original prepared
`<scene>/<scene>.mp4` + `<scene>.gt_user_c2w.npy` format remains supported.

The runner sequentially decodes from frame zero, selects `range(0,N,10)` and
verifies the decoded length. It performs no approximate seeking or GT
interpolation. Configure `configs/datasets/princeton365.yaml`; use
`configs/princeton365.yaml`.

</details>

### EuRoC

<details>
<summary>Preparation instructions and expected layout</summary>

Get the six Vicon-room sequences, calibration and Leica scans from the
[official EuRoC MAV page](https://projects.asl.ethz.ch/datasets/euroc-mav/).
Machine Hall sequences are not part of this dense setting.

From the repository root, export each official sequence:

```bash
python preprocess/euroc.py \
  --sequence /data/EuRoC/V1_01_easy \
  --output /data/euroc_processed/vicon_room1-V1_01_easy
```

The exporter reads `mav0/cam0/{data.csv,sensor.yaml,data/}` and
`mav0/state_groundtruth_estimate0/data.csv`. If the GT CSV was downloaded
separately, supply `--ground-truth /data/states/V1_01_easy.csv`. Integer
nanosecond timestamps are retained; translation uses linear interpolation and
rotation uses SLERP inside GT coverage. Outside coverage, RGB remains present
with invalid GT. Images are undistorted at the original size with alpha=0.
The output includes a calibration/association hash record.

Prepare undistorted cam0 RGB and timestamp-associated camera C2W in the scan's
registered world frame. Official `T_BS(cam0)` maps camera to body; the camera pose
is `T_WC = T_WB @ T_BS(cam0)`. Do not use body poses directly or silently invert
this extrinsic. Preserve the exact calibration/association revision used to
build the shared reference.

```text
euroc_processed/
  vicon_room1-V1_01_easy/
    camera/frames.jsonl
    rgb/...
  ...
```

Use the processed-index schema below. Set `camera_index` explicitly in
`configs/datasets/euroc.yaml` if your audited export uses another location, such
as `camera/c2w_v2/frames.jsonl`; there is no fallback between camera revisions.
Preparation checks that the selected poses agree with the reference bundle.
A changed camera revision requires regenerated references and a separate
provenance record, not an assertion of agreement with old results.

Build fixed scan-visibility references from the processed cameras and original
Leica scan. Run from the repository root:

```bash
python preprocess/dense.py --protocol euroc_dense \
  --root /data/euroc_processed --scene vicon_room1-V1_01_easy \
  --geometry /data/vicon_room1/mav0/pointcloud0/data.ply \
  --output /data/reference_bundles/euroc/vicon_room1-V1_01_easy \
  --build-reference --reference-root /data/references/euroc
```

Pass `--camera-index` if using a different camera revision, matching the dataset
YAML. Repeat for each scene, set `reference_root` to `/data/references/euroc`,
then use `configs/euroc.yaml`. This is a custom map-to-scan evaluation; see the
[validation limitation](#euroc-validation) below.

</details>

### Processed camera indexes

<details>
<summary>Preparation instructions and expected layout</summary>

CroCoDL and EuRoC adapters accept one JSON object per line:

```json
{"timestamp_us": 1000000, "image": "rgb/000000.png", "width": 640, "height": 480, "camera_model": "PINHOLE", "K": [[500,0,320],[0,500,240],[0,0,1]], "c2w": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "gt_valid": true}
```

- `image` (or `rgb`) is relative to the scene directory.
- `timestamp_us` (or `timestamp`) determines chronological order; values must be
  finite and unique. Do not independently reorder RGB and poses.
- `c2w` is the associated OpenCV camera pose in meters. For missing GT, use
  `"gt_valid": false` and `"c2w": null`; the RGB is still an inference input.
- `K` describes the already undistorted RGB and is needed when producing dense
  camera/reference bundles. GT intrinsics are never model inputs.
- `width` and `height` are the undistorted RGB dimensions in pixels, required
  for dense camera/reference bundles and constant within each dense scene.

This contract avoids a hand-written frame manifest for each evaluation, but it
is not a replacement for sensor calibration, timestamp association, registration
or vendor preprocessing. The repository supplies acquisition links, prepared-data
adapters and a reference builder; it does not claim an already distributed,
download-and-run paper reference pack for every dataset.

</details>

## Configuration System

The configuration layout follows the three-stage workflow:

```text
benchmark/
  prepare.py, run.py, evaluate.py   pipeline entrypoints
  reference.py                     dense-reference preparation
  benchmark/
    core/                          configuration, hashes and result records
    evaluation/                    pose and dense-reconstruction metrics
  datasets/
    adapters.py                    native dataset layouts and GT association
    protocols.py                   paper splits, frame selection and geometry
  methods/
    lingbot_map2.py                 native pure Online model loading and inference
  configs/
    kitti.yaml                     workspace, datasets and methods
    datasets/kitti.yaml             raw_data_root, protocol and scene selection
    methods/lingbot_map2.yaml       checkpoint, device and compilation
  envs/                            environment installation
```

The directory responsibilities follow LingBot-Map's organization. The implementation
is this repository's native pure Online pipeline. The command-line entrypoints
are unchanged.

Base configs select dataset and method configs by name. Start with a shipped base
config, set the data paths and causal checkpoint, then use that same config for
all three stages. The dataset protocol fixes frame selection, input size and
scoring settings. See [Data Preparation](#data-preparation) for acquisition links and required layouts.


## Evaluation Metrics

- ATE is per-sequence translation RMSE after one Sim(3); the dataset ATE is the
  arithmetic mean over all required sequences.
- AUC uses every valid within-sequence camera pair, float32 W2C relative
  transforms, quaternion rotation error and sign-ambiguous translation-direction
  error. It integrates one-degree histograms of the maximum of the two errors.
  Micro AUC combines pair histograms, not unweighted per-sequence AUC values.
- RPE uses adjacent valid poses (delta one frame after GT validity selection),
  with Sim(3) alignment; translation is in meters and rotation in degrees.
- Invalid GT rows are recorded and excluded from metrics. Nonfinite predictions
  are errors, including at frames without valid GT. Outputs are never silently
  shortened to improve a score.
- Plans fingerprint all selected RGB/video and GT assets. Result receipts include
  checkpoint/configuration, source identity, raw IDs, persistent IDs and hashes of
  all output arrays. Scoring rechecks hashes and rejects mixed protocols, models,
  execution settings or source versions in a dataset mean.
- Reports list missing scenes. A full dataset mean is emitted only for complete
  accepted coverage. Diagnostics and incomplete runs retain per-scene metrics.

Real-checkpoint GPU validation will accompany the pure causal checkpoint release.
Installing the package does not launch an evaluation.

## Interrupt and Resume

```bash
# Show the resolved configuration without reading data, writing files or using CUDA.
python run.py --config configs/kitti.yaml --dry-run

# One full sequence. Use the same --scene selection at every stage, or set it in YAML.
python prepare.py  --config configs/kitti.yaml --scene 04
python run.py      --config configs/kitti.yaml --scene 04
python evaluate.py --config configs/kitti.yaml --scene 04

# Reuse verified completed work; never automatically repeat formal predictions.
python prepare.py  --config configs/kitti.yaml --resume
python run.py      --config configs/kitti.yaml --resume
# Explicitly restart incomplete scenes, retaining previous attempts:
python run.py      --config configs/kitti.yaml --resume --retry-failed
python evaluate.py --config configs/kitti.yaml --resume
```

A completed scene is reused only after its plan, model/runtime/source identity
and output hashes are checked. With `--retry-failed`, incomplete scenes with
matching `ATTEMPT.json` identities
are moved to `failed_attempts/<scene>/<attempt>/` before starting that scene again.
A failed scene restarts from its first input; no KV state is inferred from partial
outputs. Verified completed scenes are skipped. Corrupted completed outputs,
changed protocols/runtime identities, or legacy folders without attempt receipts
are rejected and require a new workspace. A per-workspace lock prevents
concurrent stage writers. No `--force` option silently overwrites results. A scene subset produces
per-scene scores and an explicit missing-scene list, not a paper dataset mean.

## Output Layout

```text
outputs/benchmark/kitti/
  PREPARED.json          data configuration and plan identity
  data/                  generated input manifest and associated GT
  plan/                  frozen frame/persistence selection
  predictions/<scene>/   pose arrays, optional geometry, ATTEMPT/COMPLETE.json
  failed_attempts/       archived incomplete predictions and stale scores
  scores/<scene>.json     per-scene metrics and provenance
  scores/report.json     coverage and complete-only dataset aggregates
```

The prediction receipt's wall time includes decoding, hashing, inference, CPU
outputs and file writes. It is **not** a warmed throughput measurement. No FPS
claim should be calculated from it.

## Dense references and shared ROI

Dense manifests additionally require `"reference": "scene.reference.npz"`.
The included helper generates the previously manual `cameras.npz`,
`common-footprint.npy` and reference specification. Run it from the repository
root on the prepared RGB/camera data:

```bash
python preprocess/dense.py --protocol crocodl_dense \
  --root /data/crocodl_processed \
  --scene ARCHE_B3__ios_2023-07-12_15.40.46_000 \
  --geometry /data/meshes/ARCHE_B3/mesh_simplified.ply \
  --output /data/reference_bundles/crocodl/ARCHE_B3__ios_2023-07-12_15.40.46_000

python -m benchmark.reference \
  --spec /data/reference_bundles/crocodl/ARCHE_B3__ios_2023-07-12_15.40.46_000/reference-spec.json \
  --output /data/references/crocodl/ARCHE_B3__ios_2023-07-12_15.40.46_000.reference.npz
```

For EuRoC, use `--protocol euroc_dense`, its processed root and scene, and
`--geometry /data/vicon_room1/mav0/pointcloud0/data.ply`. If using a separate
camera revision, pass the same `--camera-index` here and in the dataset YAML.
For Oxford, use `--protocol oxford_dense` and the metric-stream root; no
`--geometry` is needed. It reads `intrinsics.txt` (`fx fy cx cy width height`),
`poses_c2w.txt` and associated raw-resolution `depth/<image-stem>.npy`, and
nearest-resizes depth to 518×388 with matching scaled intrinsics. This requires
the associated metric GT depth export; it does not generate that export from
raw LiDAR or replace it with model depth.

Repeat for the required scenes, then set `reference_root` to the directory
containing the flat `<scene>.reference.npz` files. Alternatively,
`--build-reference --reference-root /data/references/crocodl` builds the reference
directly in that flat directory. Without `--reference-root`, `--build-reference`
creates the reference inside the per-scene bundle directory. `--reference-root`
requires `--build-reference`.
Each bundle directory and reference file must be new; the shared reference root
may already contain other scenes. Existing references are never replaced.

The helper constructs the fixed shared FOV geometrically: the model's cover crop
intersected with the 504×280 width-locked canvas, plus the 518×336 cover canvas
for EuRoC. Oxford uses rows `[50,338)`. These canvases define the comparison ROI,
not additional inference runs. No predictions or confidence masks are consulted.

For an explicit camera/footprint bundle, build it **before inference** with `python -m benchmark.reference --spec
reference-spec.json --output scene.reference.npz`. A reference specification has:

```json
{
  "protocol": "crocodl_dense",
  "raw_count": 942,
  "cameras": "selected-cameras.npz",
  "footprint": "common-footprint.npy",
  "mesh": "mesh_simplified.ply"
}
```

The camera bundle contains exact selected `raw_ids`, `c2w` `[N,4,4]`, raw-camera
`K` `[N,3,3]`, and one `raw_wh` `[2]`. Intrinsics must describe the images actually
used (including any upstream undistortion). `footprint` is a Boolean scoring-grid
mask containing the frozen intersection of compared methods' observed fields of
view. Keep the same mask for every method. It is a required data asset, not an
all-image fallback or an intersection of confidence masks.

| Dataset | Reference construction | Alignment / metric settings |
| --- | --- | --- |
| CroCoDL | Raycast the official simplified NavVis mesh at 256×192; reference depth `(0.1,50)` m; apply common footprint. | Up to 250,000 evenly indexed pixel correspondences for Umeyama Sim(3), 0.02m voxels, rigid point-to-point ICP at 0.1m for 30 iterations. |
| EuRoC | Use `"scan": "data.ply"` instead of mesh. Project the original Leica scan into 256×160 views; keep the nearest positive-z point per pixel in `(0.1,20)` m and the visible scan union. | Camera-center Sim(3), 0.02m voxels, rigid ICP at 0.1m for 30 iterations. |
| Oxford | Use `"depth_grid": "depth.npy"` `[N,388,518]` and `"grid_intrinsics": "K.npy"` `[N,3,3]`, associated to the prepared metric RGB and GT poses. Shared footprint is rows `[50,338)`. | GT depth `(0.1,200)` m for metrics; GT `<80m` only for Sim(3) correspondences. 0.05m voxels and rigid ICP at 0.5m for 20 iterations. |

The implementation follows the frozen scoring order: Sim(3), voxel downsampling,
then rigid ICP on the voxelized clouds. CD is half the sum of the two mean nearest
neighbor distances, in meters. F1 is a percentage with a strict distance threshold.
No model confidence or finite-depth-range filter removes difficult predicted
points. Invalid predictions fail evaluation. Oxford nearest-resizes predicted
depth onto the GT grid and rescales intrinsics; CroCoDL/EuRoC use the explicit
pixel-center transform and nearest native-pixel lookup. Depth confidence is saved
but not used for selection.

<a id="euroc-validation"></a>
**EuRoC validation limitation:** camera-to-scan orientation, alignment and the
rasterization/ICP pipeline have not yet been validated against the complete
reference dataset. Treat these custom map-to-scan results as provisional; they
are not official EuRoC leaderboard results.

## Explicit manifests (advanced)

Obtain each dataset and its calibration/GT under the dataset's own terms. No RGB,
scanner assets, annotations or checkpoints are bundled.
All manifest paths are relative to the manifest, or absolute paths on your system.
Each scene supplies its raw frame count, ordered RGB inputs and associated C2W
ground truth in meters:

```python
import json
from pathlib import Path
import numpy as np

rgb = sorted(Path("/data/kitti/sequences/04/image_2").glob("*.png"))
rows = np.loadtxt("/data/kitti/poses/04.txt").reshape(-1, 3, 4)
poses = np.tile(np.eye(4), (len(rows), 1, 1))
poses[:, :3] = rows
Path("data").mkdir(exist_ok=True)
np.save("data/04.gt.npy", poses)
spec = {
    "protocol": "kitti_trajectory",
    "scenes": [
        {
            "name": "04",
            "raw_count": len(rgb),
            "images": [str(p) for p in rgb],
            "gt_poses": "04.gt.npy",
        }
    ],
}
Path("data/kitti.json").write_text(json.dumps(spec, indent=2))
```

This one-scene example deliberately produces a **partial report**, not the KITTI
11-scene mean. `benchmark.datasets` also converts the prepared layouts below without hand-written
image manifests. The configured workflow above invokes it automatically:

```bash
python -m benchmark.datasets --protocol kitti_trajectory \
    --root /data/kitti --output data/kitti
# Then use data/kitti/dataset.json with benchmark.prepare.
```

| Dataset | Native converter / manifest preparation |
| --- | --- |
| KITTI | `sequences/00/image_2/000000.png`, `poses/00.txt`, through 10. Matches the frozen paper loader: RGB cam2 with official cam0 GT unchanged. This convention is explicit; a stereo-extrinsic-corrected protocol would be a separate evaluation. |
| VBR | `<scene>_processed_aligned/rgb/*.png` and `processed_gt/<scene>_gt.txt`. Numeric RGB filenames may have gaps; TUM rows index **zero-based positions in the numerically sorted RGB sequence**. Arbitrary timestamps are rejected, not rounded or nearest-matched. Names include `campus_train0` and `spagna_train0`. |
| Oxford trajectory | `<scene>/images/000000.png` and raw-indexed `<scene>/poses_c2w.txt` (16 values per pose). Requires the full trajectory preparation, not the metric-only dense stream. |
| CroCoDL | Export chronologically ordered RGB paths and camera C2W from the accepted per-sequence camera metadata, preserving its scan registration and image orientation. Supply a raw-indexed GT `.npy`, or the selected-input form below. Do not infer an axis flip or transpose from filenames. |
| Tanks and Temples | Supply all ordered RGB views with the corresponding camera C2W array for Barn, Caterpillar, Church, Ignatius, Meetingroom and Truck. Convert any W2C source with a true matrix inverse before export. |
| Princeton365 | Supply the original RGB video path as `"video"` instead of `"images"`, and raw-indexed camera C2W. The runner decodes sequentially from frame zero, selects exact indices and verifies the total decoded length. No approximate video seeking is used. Preserve missing GT entries as NaNs. |
| EuRoC dense | Supply ordered cam0 RGB and associated, scan-registered camera C2W. Apply the provided camera/body calibration and accepted timestamp association during data preparation. Do not substitute body poses directly for camera poses. |

For accepted preselected assets, add `"source_indices": [0,5,10,...]` (or the
prescribed stride) and supply GT/RGB of that same length while retaining the
original `raw_count`. The indices must exactly equal the protocol selection;
trajectory protocols still require all raw indices. NPZ GT files can specify
`"gt_key": "gt_c2w"`. GT invalidity affects scoring only, never RGB selection.

The manifest boundary is intentional: timestamp association, camera/scan
registration and invalid orientation corrections depend on the dataset
preparation. The included exporters cover official EuRoC camera streams and registered CroCoDL
Capture sessions; Princeton365 can be read directly. Dataset acquisition, CroCoDL
scan registration and the Oxford metric GT depth export remain upstream steps.
Custom prepared data can use the manifest schema above.

Raw counts are checked against the frozen split. For a local smoke subset only,
set top-level `"diagnostic": true`; its report can never be marked a complete
paper evaluation. Changing the official split requires a new explicit protocol.
