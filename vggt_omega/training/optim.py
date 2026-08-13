import math

import torch


#: Modules trained from scratch when the rest of the model is a pretrained
#: checkpoint. The released VGGT-Omega weights carry no gaussian head, so these
#: start from random init while the backbone only needs nudging.
FROM_SCRATCH_PREFIXES = ("gs_dpt_head.", "gs_decoder.")


def build_param_groups(model, weight_decay, lr=None, scratch_lr_mult=1.0):
    """Parameter groups split by weight decay, and optionally by learning rate.

    ``scratch_lr_mult`` scales the LR of :data:`FROM_SCRATCH_PREFIXES` relative
    to ``lr``. One LR for the whole model is wrong when part of it is pretrained
    and part is not: at a rate that merely fine-tunes a 1B backbone, a randomly
    initialised head barely moves, and it stays under-trained for the whole run
    while the loss curve looks reasonable. Requires ``lr`` when the multiplier
    is not 1.0, since a group's LR has to be written as an absolute value.

    LambdaLR scales each group's own ``initial_lr``, so the warmup-cosine
    schedule applies on top of the per-group rate rather than flattening it.
    """
    if scratch_lr_mult != 1.0 and lr is None:
        raise ValueError("build_param_groups needs lr= to apply scratch_lr_mult")

    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scratch = name.startswith(FROM_SCRATCH_PREFIXES)
        no_decay = p.ndim <= 1 or "token" in name
        buckets.setdefault((scratch, no_decay), []).append(p)

    groups = []
    for (scratch, no_decay), params in sorted(buckets.items()):
        group = {"params": params, "weight_decay": 0.0 if no_decay else weight_decay}
        if scratch and scratch_lr_mult != 1.0:
            group["lr"] = lr * scratch_lr_mult
        groups.append(group)
    return groups


def build_warmup_cosine(optimizer, max_steps, warmup_frac=0.05, min_lr_ratio=0.0):
    warmup = max(1, int(max_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = min((step - warmup) / max(1, max_steps - warmup), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
