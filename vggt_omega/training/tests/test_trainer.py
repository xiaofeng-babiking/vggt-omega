import math
import os

import torch
from omegaconf import OmegaConf

from vggt_omega.datasets.track_util import build_tracks_by_depth
from vggt_omega.models import VGGTOmega
from vggt_omega.training.tests.conftest import _intrinsics_for_scene, _random_consistent_scene
from vggt_omega.training.trainer import Trainer

SMOKE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "train_smoke.yaml")


class SyntheticTrainData:
    """In-memory stand-in for DynamicTorchDataset: get_loader(epoch) yields
    collated batches of a textured plane scene (conftest geometry) with tracks
    from build_tracks_by_depth. Deterministic per index (batches built once)."""

    def __init__(self, num_batches=2, B=2, S=2, H=64, W=64, track_num=16):
        torch.manual_seed(1234)
        self.batches = [
            self._build_batch(idx, B, S, H, W, track_num) for idx in range(num_batches)
        ]

    @staticmethod
    def _build_batch(idx, B, S, H, W, track_num):
        ext, dep, wp, mask = _random_consistent_scene(B=B, S=S, H=H, W=W, seed=idx)
        K = _intrinsics_for_scene(B, S, H, W)
        g = torch.Generator().manual_seed(1000 + idx)
        images = torch.rand(B, S, 3, H, W, generator=g)
        tracks, vis, pos = [], [], []
        for b in range(B):
            t, v, p = build_tracks_by_depth(
                ext[b], K[b], wp[b], dep[b], mask[b], images[b],
                target_track_num=track_num, neg_ratio=0.25,
            )
            tracks.append(t)
            vis.append(v)
            pos.append(p)
        return {
            "images": images,
            "depths": dep,
            "extrinsics": ext,
            "intrinsics": K,
            "world_points": wp,
            "point_masks": mask,
            "ids": torch.arange(S).expand(B, S).clone(),
            "tracks": torch.stack(tracks),
            "track_vis_mask": torch.stack(vis),
            "track_positive_mask": torch.stack(pos),
            "seq_name": [f"synthplane_{idx}_{b}" for b in range(B)],
            "modalities": [["depths", "extrinsics"]] * B,
        }

    def get_loader(self, epoch):
        return list(self.batches)


def _smoke_cfg(tmp_path):
    cfg = OmegaConf.load(SMOKE_CONFIG)
    cfg.run.max_steps = 12
    cfg.run.log_interval = 1
    cfg.run.ckpt_interval = 6
    cfg.run.output_dir = str(tmp_path)
    cfg.model.checkpoint = None
    cfg.model.embed_dim = 64
    return cfg


def test_train_steps_decrease_loss_and_log(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    t = Trainer(cfg, data_override=SyntheticTrainData())
    t.fit()
    assert t.global_step == 12
    first, last = t.loss_history[0], t.loss_history[-1]
    assert last < first
    assert all(math.isfinite(x) for x in t.loss_history)
    event_files = list((tmp_path / "tb").glob("events.out.tfevents.*"))
    assert event_files, "tensorboard events must be written"
    ckpts = sorted(tmp_path.glob("model_step*.pt"))
    assert ckpts and isinstance(torch.load(ckpts[-1], map_location="cpu"), dict)
    fresh = VGGTOmega(embed_dim=64)
    fresh.load_state_dict(torch.load(ckpts[-1], map_location="cpu"))


def test_resume_restores_step_and_optimizer(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    cfg.run.max_steps = 6
    t1 = Trainer(cfg, data_override=SyntheticTrainData())
    t1.fit()
    assert t1.global_step == 6
    sidecar = tmp_path / "trainer_step000006.pt"
    assert sidecar.exists()

    cfg2 = _smoke_cfg(tmp_path)
    t2 = Trainer(cfg2, data_override=SyntheticTrainData())
    t2.resume(str(sidecar))
    assert t2.global_step == 6
    assert len(t2.optimizer.state) > 0
    assert any("exp_avg" in s for s in t2.optimizer.state.values())
    t2.fit()
    assert t2.global_step == 12
    assert len(t2.loss_history) == 6
    assert all(math.isfinite(x) for x in t2.loss_history)
