<div align="center">
<h1>VGGT-&Omega;</h1>

<a href="http://vggt-omega.github.io/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2605.15195" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/arXiv-2605.15195-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/spaces/facebook/vggt-omega"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>

<p>
  <span class="author"><a href="https://jytime.github.io/">Jianyuan Wang</a><sup>1,2</sup></span>
  <span class="author"><a href="https://silent-chen.github.io/">Minghao Chen</a><sup>1</sup></span>
  <span class="author"><a href="https://scholar.google.com/citations?user=FUDsZkEAAAAJ&amp;hl=zh-CN">Shangzhan Zhang</a><sup>1</sup></span>
  <span class="author"><a href="https://nikitakaraevv.github.io/">Nikita Karaev</a><sup>1</sup></span>
  <br>
  <span class="author"><a href="https://demuc.de/">Johannes Schönberger</a><sup>2</sup></span>
  <span class="author"><a href="https://scholar.google.com/citations?user=IJidh-UAAAAJ&amp;hl=fr">Patrick Labatut</a><sup>2</sup></span>
  <span class="author"><a href="https://scholar.google.com/citations?user=lJ_oh2EAAAAJ&amp;hl=en">Piotr Bojanowski</a><sup>2</sup></span>
  <span class="author"><a href="https://d-novotny.github.io/">David Novotny</a></span>
  <br>
  <span class="author"><a href="https://www.robots.ox.ac.uk/~vedaldi/">Andrea Vedaldi</a><sup>1,2</sup></span>
  <span class="author"><a href="https://chrirupp.github.io/">Christian Rupprecht</a><sup>1</sup></span>
</p>

**<sup>1</sup>[Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/)**; **<sup>2</sup>[Meta AI](https://ai.facebook.com/research/)**
</div>

## Pretrained models

Before using the models, please request access to the checkpoints [here](https://huggingface.co/facebook/VGGT-Omega). Once your request is approved, you can download the checkpoints. Please note that access requests are reviewed by an automated process based on the information provided in the request.

| Model | Resolution | Text alignment | Download |
| :--- | :--- | :--- | :--- |
| `VGGT-Omega-1B-512` | 512 | No | [Link](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt) |
| `VGGT-Omega-1B-256-Text-Alignment` | 256 | Yes | [Link](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_256_text.pt) |

The authors are not involved in the review process and cannot approve or reject individual applications. However, the [🤗 Hugging Face demo](https://huggingface.co/spaces/facebook/vggt-omega) is available to everyone.


## Quick Start

First, clone this repository and install the dependencies:

```bash
git clone git@github.com:facebookresearch/vggt-omega.git
cd vggt-omega
pip install -r requirements.txt
pip install -e .
```


Now, try the model with a few lines of code:

```python
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

checkpoint_path = "path/to/vggt_omega_1b_512.pt"
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]

model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = load_and_preprocess_images(image_names, image_resolution=512).to("cuda")

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
camera_and_register_tokens = predictions["camera_and_register_tokens"]
camera_tokens = camera_and_register_tokens[:, :, :1]
registers = camera_and_register_tokens[:, :, 1:]
```

For the text-aligned checkpoint, use `VGGTOmega(enable_alignment=True)` with `image_resolution=256` and read `predictions["text_alignment_embedding"]`.


## Interactive Demo

Install the demo dependencies:

```bash
pip install -r requirements_demo.txt
```

Launch the Gradio demo with a local checkpoint path:

```bash
python demo_gradio.py \
  --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
  --image-resolution 512
```

The demo accepts uploaded images or a video, runs camera and depth inference,
and visualizes the depth-unprojected point cloud and predicted cameras as a GLB
scene.

## Dataset Visualization (Rerun)

Inspect a dataset sequence — RGB, depth, camera frusta, the world point cloud,
and the camera trajectory — in [Rerun](https://rerun.io), with every modality
placed on its real per-frame **timestamp** timeline so you can scrub by capture
time. It runs **headless**: write a `.rrd` per sequence (or stream live) and view
it in a browser via `rerun --serve-web`.

Install the optional viz dependency (adds `rerun-sdk`, which also provides the
`rerun` CLI):

```bash
pip install -e '.[viz]'
```

Point the loader at your data by editing `TUM_DIR` in
[`vggt_omega/datasets/config/tum.yaml`](./vggt_omega/datasets/config/tum.yaml).
This is the same per-dataset `--configure` file inference uses, so what you
visualize is exactly what training/inference tensorizes.

### Write `.rrd` files, then serve them

```bash
# one .rrd per TUM sequence found under TUM_DIR
python -m vggt_omega.datasets.adapters \
  --configure vggt_omega/datasets/config/tum.yaml \
  --out rerun_out --num-frames 90 --point-stride 8

# serve the web viewer; open the printed http://<host>:9090 in a browser
rerun --serve-web rerun_out/*.rrd
```

On a remote/headless host, forward the web port over SSH
(`ssh -L 9090:localhost:9090 <host>`) and open `http://localhost:9090`.

### Stream live (no files)

```bash
# A) into a running web viewer — open the browser FIRST (it forwards live)
rerun --serve-web &
python -m vggt_omega.datasets.adapters \
  --configure vggt_omega/datasets/config/tum.yaml --connect

# B) host a buffering server in-process; attach a viewer whenever, then Ctrl-C
python -m vggt_omega.datasets.adapters \
  --configure vggt_omega/datasets/config/tum.yaml --serve
```

Useful flags: `--seq-index N` (one sequence; default = all), `--num-frames K`
(`<=0` = all frames, evenly spaced; lower it for large sequences),
`--point-stride S` (subsample the world cloud — higher = lighter output), and
`--accumulate` (keep every frame's cloud instead of replacing it per frame).

The adapter is modality-driven, so the same commands work for the 7-Scenes loader
(`vggt_omega/datasets/config/` + `--configure`) and any future vendor — it renders
whatever modalities a sample declares.

### Wrap a dataset for a train / test pipeline (`RerunDataset`)

`RerunDataset` wraps **any** VGGT-Omega dataset into a transparent, torch-style
logging dataset: it forwards every method to the inner dataset, and each sample
you fetch — via `ds[idx]` (the training/DataLoader path) or `ds.get_sample(...)`
(the ordered eval/inference path) — passes through **unchanged** while being
logged to Rerun as a side effect. Drop it into any pipeline for free 3D
inspection, with no changes to your data flow:

```python
from torch.utils.data import DataLoader
from vggt_omega.datasets.adapters import RerunDataset

# A) one .rrd per sequence — offline, fork-safe with DataLoader workers
ds = RerunDataset(base_dataset, out_dir="rerun_out", point_stride=8)
for sample in DataLoader(ds, batch_size=None, num_workers=4):
    train_step(sample)                      # each sample also written to rerun_out/

# B) stream every sample into one live recording (single process)
import rerun as rr
rr.init("vggt", spawn=True)                 # or pass recording=rr.RecordingStream(...)
ds = RerunDataset(base_dataset)             # logs to the current recording
for sample in ds:                           # scrub the `sample` timeline in the viewer
    ...

# C) drop-in for the eval/inference path (same signature as the dataset)
viz = RerunDataset(dataset, out_dir="rerun_out")
sample = viz.get_sample(seq_index=0, ids=[0, 10, 20], aspect_ratio=0.75)
```

Use `out_dir=` for DataLoader workers (`num_workers > 0`); the `recording=` /
current-recording modes hold a live stream and are single-process only.

Or log a single already-loaded sample with the lower-level primitives:

```python
from vggt_omega.datasets.adapters import sample_to_rrd  # or: log_sample(sample, recording)

sample = dataset.get_sample(seq_index=0, ids=[0, 10, 20], aspect_ratio=0.75)
sample_to_rrd(sample, "seq.rrd")   # then: rerun --serve-web seq.rrd
```

## Runtime and GPU Memory

We benchmark the end-to-end peak GPU memory usage of `VGGT-Omega-1B-512` on a
single NVIDIA A100 GPU with 624x416 input images. The measurement covers the full
inference program, from loading the model weights onto the GPU through the
forward pass, so it includes both the memory needed to store the model itself
and the memory used by inference activations and buffers. In other words, a GPU
with at least the listed available memory is able to run the corresponding
number of input frames under this setup.

| **Input Frames** | 1 | 10 | 25 | 50 | 100 | 200 | 300 | 400 | 500 |
|:----------------:|:-:|:--:|:--:|:--:|:---:|:---:|:---:|:---:|:---:|
| **Peak Memory (GB)** | 6.02 | 6.67 | 7.80 | 9.66 | 13.37 | 20.82 | 28.26 | 35.71 | 43.15 |

The benchmark uses [`load_and_preprocess_images`](./vggt_omega/utils/load_fn.py)
with the default `mode="balanced"` and `image_resolution=512`. For these roughly
3:2 landscape images, this produces 624x416 inputs. You can set
`mode="max_size"` to resize the longest side to 512 instead; for the same aspect
ratio, this gives about 512x336 inputs and uses less GPU memory.

### Estimating peak memory

Peak usage is well approximated by a fixed floor plus a term **linear in the total
token count**. At patch size 16, an `H×W` input produces `P = (H÷16)·(W÷16)` patch
tokens per frame, so for `N` frames:

```
peak_GB  ≈  5.9  +  7.3e-5 · N · P
```

**Where the constants come from.** Both are a least-squares fit to the table above,
which is linear to within ~1%:

- **`5.9 GB` — the floor** (independent of frame count): the model weights plus
  runtime overhead. The `_1b_512` checkpoint is `1.144 B` parameters stored in
  fp32, so the weights occupy `≈ 4.6 GB` on the GPU, and the CUDA / cuDNN / cuBLAS
  context and allocator overhead add `≈ 1.3 GB`. Loading the weights in bf16 would
  drop the floor to `≈ 3.6 GB`.
- **`7.3e-5 GB ≈ 73 KB` per patch token — the slope**: the marginal activation per
  token, dominated by the four cached multi-layer features the heads consume (each
  `2·embed_dim` wide, kept for all frames) plus the per-pixel depth/conf outputs,
  with the caching allocator's reservation headroom folded in. These are
  `embed_dim`-wide activations (`73 KB ≈ 18 × embed_dim × 4 B`), so the slope scales
  with `embed_dim`; resolution only changes `P`, the number of tokens.

The term is linear in `N` (not quadratic) because inference runs under
`torch.inference_mode()`, which frees activations as they are consumed, and
`scaled_dot_product_attention` uses the memory-efficient backend, so attention
memory stays linear even though its compute is `O(N²)`. The per-token cost is
essentially resolution-independent (`7.3e-5` at both 624x416 and 640x480), so peak
is set by the total token count `N·P` — which is why the estimate takes only frame
count and resolution as inputs.

Worked examples: 624x416 (`P = 1014`) at 500 frames → `≈ 42.9 GB` (43.15 measured);
native 640x480 (`P = 1200`) → `≈ 99 GB` at 1069 frames (OOM on an 80 GB GPU, as
observed). The constants are calibrated for this 1B model with fp32 weights on an A100-class GPU
and shift with dtype, attention backend, or the alignment head enabled; the model
covers memory only, not the `O(N²)` runtime.

## License

See the [LICENSE](./LICENSE) file for details about the license under which
this code is made available.

[^release]: This Release is intended to support the open source research community.

```bibtex
@misc{wang2026vggtomega,
      title={VGGT-$\Omega$}, 
      author={Jianyuan Wang and Minghao Chen and Shangzhan Zhang and Nikita Karaev and Johannes Schönberger and Patrick Labatut and Piotr Bojanowski and David Novotny and Andrea Vedaldi and Christian Rupprecht},
      year={2026},
      eprint={2605.15195},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.15195}, 
}
```
