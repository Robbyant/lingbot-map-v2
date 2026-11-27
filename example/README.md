# Example sequences

Five ordered image sequences for Online streaming reconstruction. The four hosted
archives contain the full example inputs. KITTI 08 uses a fixed selection of
321 original color frames from the odometry benchmark.

| Sequence | Input frames | Persistent frames with the documented settings |
| --- | ---: | ---: |
| `courthouse` (Courthouse) | 286 | 286 |
| `university` (University) | 324 | 320 |
| `bund` (The Bund / 外滩) | 480 | 320 |
| `apartment` (Apartment / 公寓) | 480 | 320 |
| `kitti08` (KITTI Odometry 08) | 321 | 321 |

## Download the four hosted scenes

The **1,570 images** total **985,781,941 bytes (approximately 0.918 GiB)**.
They are distributed independently in the
[Hugging Face example dataset](https://huggingface.co/datasets/robbyant/lingbot-map-v2-examples).
Image files are not included in the GitHub code repository. The dataset contains
four uncompressed TAR archives, one per scene. Extracting an archive restores
the original images under `example/<scene>/` in manifest order.

Run from the code repository root:

```bash
pip install -U huggingface_hub
# While the dataset is private, first log in with an authorized account:
hf auth login
hf download robbyant/lingbot-map-v2-examples --repo-type dataset \
    courthouse.tar university.tar bund.tar apartment.tar \
    manifest.json --local-dir example
(cd example && for scene in courthouse university bund apartment; do tar -xf "$scene.tar" || exit 1; done)
```

The `archives` section of `manifest.json` records the SHA-256 of each TAR.
Each scene lists its images in order with their original dimensions, byte sizes
and SHA-256. To verify the four downloaded archives, run this from `example/`:

```bash
python - <<'PYTHON'
import hashlib
import json
from pathlib import Path

for name, archive in json.loads(Path("manifest.json").read_text())["archives"].items():
    digest = hashlib.sha256()
    with open(name, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != archive["sha256"]:
        raise SystemExit(f"Checksum mismatch: {name}")
    print(f"Verified {name}")
PYTHON
```

## Run

Install the visualization dependencies and download the Base Streaming model
as described in the main README, then run:

```bash
python demo.py \
    --checkpoint checkpoints/base-streaming/model_weights.pt \
    --config checkpoints/base-streaming/config.json \
    --images example/courthouse \
    --width 512 --height 288 \
    --mask-sky
```

For the four hosted scenes, replace `courthouse` with the chosen scene name. By default,
`demo.py` caps persistent KV-cache keyframes to `320` (`--num-keyframes 320`): sequences
with up to `320` frames keep every frame, while longer sequences keep the first eight
initialization frames plus `312` uniformly spaced frames. All input frames receive
predictions without non-keyframes entering the KV cache. Inference completes before the
viewer starts.
The viewer uses point size `0.00001`, camera size `0.01`, visibility threshold
`1.5` and downsample factor `10` by default. The checkpoint is downloaded
separately.

## KITTI 08

Download **odometry data set (color)** from the [official KITTI odometry page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) using your registered account. From the code repository root, pass the original ZIP or the extracted `dataset/sequences/08/image_2` directory to the helper:

```bash
python example/prepare_kitti08.py /path/to/data_odometry_color.zip
python demo.py --checkpoint checkpoints/base-streaming/model_weights.pt \
    --config checkpoints/base-streaming/config.json --images example/kitti08 \
    --width 512 --height 288 --keyframe-interval 1 --depth-stride 1
```

The helper copies the 321 PNGs listed in `manifest.json`, checks each SHA-256, and preserves the original six-digit frame names. Natural filename order reproduces the tested Uniform320 selection from the 4,071-frame sequence. Use `--keyframe-interval 1` because these inputs are already keyframes. All 321 produce camera poses and dense depth. With `--depth-stride 1`, the viewer projects every frame into the point cloud.

The private HF dataset also supplies the helper and manifest independently:

```bash
hf download robbyant/lingbot-map-v2-examples --repo-type dataset \
    prepare_kitti08.py manifest.json --local-dir example
```

KITTI imagery is credited to Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun (*Vision meets Robotics: The KITTI Dataset*, 2013). KITTI lists [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) for its datasets. Obtain the original images through the [official download process](https://www.cvlibs.net/datasets/kitti/user_login.php) under the [KITTI terms of service](https://www.cvlibs.net/datasets/kitti/terms_of_service.php).

## Sources

- `courthouse` and `university` are reproduced from the
  [LingBot-Map examples](https://github.com/Robbyant/lingbot-map/tree/b9dd1b1de535e6d36e777fb42b8433c781bfd783/example)
  at commit `b9dd1b1de535e6d36e777fb42b8433c781bfd783`. Refer to the upstream
  project and the code repository's third-party notices for attribution.
- `bund` and `apartment` are example sequences provided by the Robbyant Team.
  Each contains 480 uniformly selected frames from its source video, saved at
  native resolution as JPEG images. The source video files are not included.
