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
