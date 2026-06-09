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

## Distributed Inference

`inference.py` runs a whole sequence on a single GPU, so the longest sequence it
can handle is capped by the memory curve above (≈ 1069 frames at 640×480 on an
80 GB GPU). For longer sequences, `distributed_inference.py` shards the frames
across multiple GPUs with **context parallelism**: each rank embeds only its slice
of the frames, and the cross-frame ("global") attention is computed jointly across
ranks. Both the per-GPU activation memory and the `O((N·P)²)` global-attention
compute are split across the GPUs — so beyond the fixed model floor (the weights,
replicated on every rank), `G` GPUs run roughly `G×` longer sequences. The result
is **mathematically equivalent** to a single-GPU run (only floating-point
reduction order differs), and the released checkpoint is loaded unchanged.

### Launch

It mirrors `inference.py` — the same dataset-driven contract and the same
`--configure` / `--checkpoint` / `--output_root` flags (frame count and resolution
still come from the configure's `inference` block) — but it is started with
`torchrun`, one rank per GPU, over the NCCL backend:

```bash
# Single node, 8 GPUs
torchrun --standalone --nproc_per_node=8 distributed_inference.py \
  --configure vggt_omega/datasets/config/tum.yaml \
  --checkpoint /jfs/jing.feng/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt \
  --output_root outputs \
  --cp_strategy all_gather_kv
```

```bash
# Multi-node (e.g. 2 nodes × 8 GPUs); run on every node with a shared rendezvous
torchrun --nnodes=2 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE_IP:29500 \
  distributed_inference.py \
  --configure vggt_omega/datasets/config/tum.yaml \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --cp_strategy ring
```

The single-GPU `inference.py` is left unchanged; use it whenever a sequence
already fits on one GPU.

### Attention strategy (`--cp_strategy`)

The cross-rank global attention has two interchangeable, numerically-exact
implementations:

| `--cp_strategy` | How it works | Best for |
| :--- | :--- | :--- |
| `all_gather_kv` (default) | All-gathers K/V to every rank, then computes local-query × global-KV attention | A single node — the gathered K/V ride the fast NVLink / NVSwitch links |
| `ring` | Rotates K/V blocks around the ranks with online-softmax accumulation; never materializes the full K/V | Multiple nodes or extremely long sequences — minimal, overlap-friendly communication and the lowest per-rank memory |

### What is communicated

Frames live in the batch dimension, so the DINOv2 patch embedding, the per-frame
attention blocks, and the entire dense depth/confidence head run **independently
per rank with no communication**. The only cross-GPU exchange is the global/cross
attention — the aggregator's inter-frame "global" and "register" blocks and the
camera head's trunk — i.e. exactly the part that must mix information across all
frames.

### Outputs

Results land under `--output_root/<sequence>/` as in single-GPU runs, written
cooperatively across ranks:

- **Depth / confidence PNGs** are written per rank, named by the **global** frame
  index (`depth/frame_0000.png`, …), so the on-disk layout matches a single-GPU run.
- **Point cloud** is written as one partial PLY per rank (`pointcloud_rank{r}.ply`).
- **Metrics** (`metrics/metrics.json`, written by rank 0): mono-depth metrics
  (Abs Rel / δ) are reduced across ranks (frame-count weighted), and camera-pose
  ATE / RPE is scored on rank 0 over the trajectory gathered from every rank.

### Profiling

**Step 1 — run with `--profile`.** This profiles the **first sequence's forward**
with [`torch.profiler`](https://pytorch.org/docs/stable/profiler.html) (after a
warmup pass), logs the rank-0 operator table, and writes one Chrome trace per rank:

```bash
torchrun --standalone --nproc_per_node=8 distributed_inference.py \
  --configure vggt_omega/datasets/config/tum.yaml \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --cp_strategy all_gather_kv \
  --profile
```

Tip: cap `num_frames` in the configure's `inference` block to a representative
value (e.g. 128–256) so the trace stays small and the run finishes quickly.

**Step 2 — collect the expected output.** A `--profile` run produces, for the
first sequence (under `--output_root/<sequence_name>/`):

- **One Chrome trace per rank**: `trace_rank0.json`, `trace_rank1.json`, …,
  `trace_rank{G-1}.json`, where `G = --nproc_per_node`. Each is tens of MB (≈ 18 MB
  at 256 frames on 8 GPUs). The ranks are frame-shards of the same forward, so any
  one is representative; comparing them surfaces load imbalance from uneven shards.
- **The rank-0 operator table** (sorted by CUDA time) printed to the log/stderr.

Independently of `--profile`, **every** forward logs each rank's peak memory —
`[rank r] N local frames | peak GPU mem X GB` — handy for capacity planning and
the frames-per-GPU ceiling.

**Step 3 — read the op table.** The two rows that matter for context-parallel inference are:

- `nccl:all_gather` / `ncclDevKernel_AllGather…` — the cross-GPU **communication**
  (the per-block K/V exchange).
- `…flash_attention…` / `aten::scaled_dot_product_attention` — the global + per-frame
  **attention compute**.

If the all-gather rows dominate, you are communication-bound — the K/V exchange is
serialized with compute under `all_gather_kv`, so on a slow interconnect (PCIe,
multi-node) try `--cp_strategy ring`, use fewer GPUs per sequence, or move to
NVLink/SXM hardware. If the attention rows dominate, you are at the `O((N·P)²)`
attention wall, which is inherent to exact global attention. (Ignore the high
`aten::copy_` *CPU* percentage — that is the CPU blocking on the GPU, not real work;
read the **Self CUDA** column.)

**Step 4 — open the timeline.** Load any `trace_rank{r}.json` in `chrome://tracing`
or [perfetto.dev](https://ui.perfetto.dev) to see the gather ↔ attention timeline
(e.g. whether the K/V all-gather is serialized with compute). For a full multi-GPU
system timeline, wrap the launch in
[Nsight Systems](https://developer.nvidia.com/nsight-systems):
`nsys profile -o cp_infer --trace=cuda,nvtx,osrt torchrun … distributed_inference.py …`.

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
