"""Learning-rate schedule in optimiser steps."""

import math


def warmup_cosine_lr(
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_steps: int,
    start_lr: float = 0.0,
    final_lr: float = 1e-6,
) -> float:
    """Linear warmup from ``start_lr`` to ``base_lr``, then cosine decay to ``final_lr``.

    Upstream LPL uses the same shape (pl_bolts ``LinearWarmupCosineAnnealingLR``), stepped per
    epoch; here it is stepped per optimiser step so every condition shares it exactly.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return start_lr + (base_lr - start_lr) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * min(1.0, progress)))
