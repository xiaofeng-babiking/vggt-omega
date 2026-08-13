import torch

from vggt_omega.models.vggt_omega import VGGTOmega
from vggt_omega.training.optim import build_param_groups, build_warmup_cosine


def test_param_groups_no_decay_for_1d_and_tokens():
    m = VGGTOmega(embed_dim=64)
    groups = build_param_groups(m, weight_decay=0.05)
    assert {g["weight_decay"] for g in groups} == {0.0, 0.05}
    name_by_id = {id(p): n for n, p in m.named_parameters()}
    groups_by_wd = {g["weight_decay"]: [name_by_id[id(p)] for p in g["params"]] for g in groups}
    no_decay_names = groups_by_wd[0.0]
    assert any("camera_token" in n for n in no_decay_names)
    assert any(n.endswith("gamma") for n in no_decay_names)       # LayerScale
    assert all(not n.endswith("weight") or "norm" in n.lower() for n in no_decay_names if ".attn." in n)
    total = sum(len(g["params"]) for g in groups)
    assert total == sum(1 for p in m.parameters() if p.requires_grad)


def test_warmup_cosine_shape():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
    sched = build_warmup_cosine(opt, max_steps=1000, warmup_frac=0.05)
    lrs = []
    for _ in range(1000):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < 1e-5                       # starts near 0
    assert abs(max(lrs) - 2e-4) < 1e-9         # peaks at peak_lr
    assert lrs.index(max(lrs)) == 49           # peak at end of 5% warmup
    assert lrs[-1] < 1e-6                      # cosine decays to ~0
    assert all(b <= a + 1e-12 for a, b in zip(lrs[50:], lrs[51:]))  # monotone decay after warmup


# --- from-scratch LR group ----------------------------------------------------
import pytest  # noqa: E402
from vggt_omega.training.optim import FROM_SCRATCH_PREFIXES  # noqa: E402


class _TwoPartModel(torch.nn.Module):
    """A pretrained-looking backbone plus a from-scratch gaussian head."""

    def __init__(self):
        super().__init__()
        self.aggregator = torch.nn.Linear(4, 4)
        self.gs_dpt_head = torch.nn.Linear(4, 4)


def _lookup(groups, param):
    return next(g for g in groups if any(p is param for p in g["params"]))


def test_param_groups_keep_one_lr_when_the_multiplier_is_one():
    """Default behaviour is unchanged: no group carries an explicit lr, so every
    parameter runs at the optimizer's."""
    groups = build_param_groups(_TwoPartModel(), 0.05, lr=1e-4)
    assert all("lr" not in g for g in groups)


def test_param_groups_raise_scratch_modules_without_touching_the_backbone():
    model = _TwoPartModel()
    groups = build_param_groups(model, 0.05, lr=1e-4, scratch_lr_mult=50.0)
    assert _lookup(groups, model.gs_dpt_head.weight)["lr"] == pytest.approx(5e-3)
    # The backbone must NOT get an explicit lr -- it inherits the optimizer's.
    assert "lr" not in _lookup(groups, model.aggregator.weight)


def test_param_groups_still_split_weight_decay_within_each_lr_group():
    model = _TwoPartModel()
    groups = build_param_groups(model, 0.05, lr=1e-4, scratch_lr_mult=50.0)
    # 2 lr regimes x 2 decay regimes, and biases (ndim 1) never decay.
    assert len(groups) == 4
    assert _lookup(groups, model.gs_dpt_head.bias)["weight_decay"] == 0.0
    assert _lookup(groups, model.gs_dpt_head.weight)["weight_decay"] == 0.05
    assert _lookup(groups, model.gs_dpt_head.bias)["lr"] == pytest.approx(5e-3)


def test_param_groups_reject_a_multiplier_without_a_base_lr():
    with pytest.raises(ValueError, match="lr="):
        build_param_groups(_TwoPartModel(), 0.05, scratch_lr_mult=10.0)


def test_from_scratch_prefixes_match_the_real_gaussian_modules():
    """The prefixes are matched against named_parameters(), so a rename in the
    model would silently drop the head back to the backbone's LR."""
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(embed_dim=64, enable_3dgs=True, gs_sh_degree=1)
    names = [n for n, _ in model.named_parameters()]
    assert any(n.startswith(FROM_SCRATCH_PREFIXES) for n in names)
