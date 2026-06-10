import math

import torch


def build_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "token" in name else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def build_warmup_cosine(optimizer, max_steps, warmup_frac=0.05, min_lr_ratio=0.0):
    warmup = max(1, int(max_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = min((step - warmup) / max(1, max_steps - warmup), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
