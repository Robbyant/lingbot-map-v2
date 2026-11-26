# Third-party notices

Original code by The Robbyant Team Authors is licensed under CC BY-NC 4.0,
as provided in the root [LICENSE.txt](../LICENSE.txt), and is copyright 2026
The Robbyant Team Authors. This default does not replace the upstream licenses
of third-party files or their adaptations.

The encoder, transformer layers, DPT head, positional embeddings and pose utilities
include adaptations of the following public projects:

- [DINOv3](https://github.com/facebookresearch/dinov3) and
  [DINOv2](https://github.com/facebookresearch/dinov2) by Meta.
- [VGGT](https://github.com/facebookresearch/vggt) by Meta and the University of Oxford.
- The DINOv3 encoder adaptation in
  [VGGT-Omega](https://github.com/facebookresearch/vggt-omega).

Local adaptations provide online inference, checkpoint-compatible model assembly
and paged attention. Original copyright notices are retained in the source files.

Third-party files retain their upstream notices and license texts:

- CC BY-NC 4.0: root [LICENSE.txt](../LICENSE.txt).
- Apache-2.0 (including `lingbot_map2/layers/layer_scale.py`,
  `lingbot_map2/layers/mlp.py`, and `lingbot_map2/vis/*`):
  [LICENSES/Apache-2.0.txt](Apache-2.0.txt).
- DINOv3 License (`lingbot_map2/layers/dinov3/*`, adapted from
  [DINOv3](https://github.com/facebookresearch/dinov3)):
  [LICENSES/DINOv3-License.txt](DINOv3-License.txt).
- FAIR Noncommercial Research License v1 (`lingbot_map2/layers/dinov3/*`,
  encoder adaptation from
  [VGGT-Omega](https://github.com/facebookresearch/vggt-omega)):
  [LICENSES/FAIR-Noncommercial-Research-License-v1.txt](FAIR-Noncommercial-Research-License-v1.txt).

The Robbyant Team licenses its original Base Streaming model weights under
[CC BY-NC 4.0](../LICENSE.txt). The upstream terms above continue to apply to
third-party components.

Visualization controls, point cloud/camera display, sky segmentation, GLB export
and color utilities in `lingbot_map2/vis/viewer.py`, `lingbot_map2/vis/glb_export.py`,
`lingbot_map2/vis/utils.py` and `lingbot_map2/vis/sky_segmentation.py` are adapted
from [LingBot-Map](https://github.com/Robbyant/lingbot-map/tree/b9dd1b1de535e6d36e777fb42b8433c781bfd783/lingbot_map/vis),
distributed under Apache-2.0. The original Meta copyright notices are retained;
the Apache-2.0 license is included in `LICENSES/Apache-2.0.txt`. These adaptations
connect the visualization layer to Online streaming outputs and saved-prediction
playback.
The optional skyseg ONNX weights are downloaded separately from
[JianyuanWang/skyseg](https://huggingface.co/JianyuanWang/skyseg) and are not bundled
with this source release.

The separately distributed `courthouse` and `university` example image sequences
are reproduced from the example inputs of the same LingBot-Map revision linked
above. Their original encoded contents are retained. See
[`example/README.md`](../example/README.md) for the source link and input inventory.
