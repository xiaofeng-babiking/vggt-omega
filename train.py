"""End-to-end supervised training for VGGT-Omega (arXiv 2605.15195 recipe).

Loads a training config (recipe + 16-vendor data mixture) with OmegaConf and
hands it to :class:`vggt_omega.training.trainer.Trainer`: four-term loss
(camera/depth/point/match), AdamW + warmup-cosine, bf16 inside the model,
gradient checkpointing, DDP via torchrun, TensorBoard under
``<out_root>/<run_name>/tb``, bare-state-dict checkpoints (loadable by
inference.py / demo / distributed_inference.py) with a trainer sidecar.

Usage::

    .venv/bin/torchrun --standalone --nproc_per_node=8 train.py \
        --config vggt_omega/training/config/train_default.yaml
    .venv/bin/python train.py --config vggt_omega/training/config/train_smoke.yaml  # single device

``--help`` lists every flag.
"""

import os
import sys
import time

import gflags
from omegaconf import OmegaConf

from vggt_omega.training.trainer import Trainer

FLAGS = gflags.FLAGS
# Flag names must stay DISJOINT from inference.py's (checkpoint, configure,
# output_root, conf_percentile, max_points): Trainer._validate lazily imports
# inference, which registers its flags on the same gflags singleton — any
# shared name raises DuplicateFlagError.
gflags.DEFINE_string(
    "config",
    "vggt_omega/training/config/train_default.yaml",
    "Training config YAML (recipe + data mixture).",
)
gflags.DEFINE_string("out_root", "outputs", "Root for run dirs (gitignored).")
gflags.DEFINE_string("run_name", None, "Run dir name; default train_<UTC timestamp>.")
gflags.DEFINE_string("resume", None, "Path to a trainer_step*.pt sidecar to resume from.")
gflags.DEFINE_string(
    "init_checkpoint", None, "Override cfg.model.checkpoint (model init weights)."
)


def main():
    cfg = OmegaConf.load(FLAGS.config)
    if FLAGS.init_checkpoint:
        cfg.model.checkpoint = FLAGS.init_checkpoint
    run_name = FLAGS.run_name or time.strftime("train_%Y%m%d_%H%M%S", time.gmtime())
    output_dir = os.path.join(FLAGS.out_root, run_name)
    cfg.run.output_dir = output_dir
    if int(os.environ.get("RANK", "0")) == 0:
        os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    trainer = Trainer(cfg)
    if FLAGS.resume:
        trainer.resume(FLAGS.resume)
    trainer.fit()


if __name__ == "__main__":
    try:
        FLAGS(sys.argv)
    except gflags.FlagsError as err:
        sys.exit(f"{err}\nUsage: {sys.argv[0]} --config <yaml>\nUse --help for the full flag list.")
    main()
