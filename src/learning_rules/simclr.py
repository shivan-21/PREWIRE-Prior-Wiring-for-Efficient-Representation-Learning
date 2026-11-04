"""SimCLR baseline: end-to-end VGG-11 with a projection head and the NT-Xent contrastive loss.

Chen et al. (2020). The encoder is the shared backbone (``src/models/vgg.py``) trained end to end;
the head is the standard 2-layer MLP and is discarded after pre-training. Pairs come from the same
augmentation generator as LPL (2-frame, rho = 0 clips), so only the learning rule differs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vgg import FEATURE_DIM

TEMPERATURE = 0.5
HIDDEN_DIM = 2048
PROJECTION_DIM = 128


class ProjectionHead(nn.Module):
    """MLP projection head: Linear -> ReLU -> Linear."""

    def __init__(
        self, in_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM, out_dim: int = PROJECTION_DIM
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(
    h1: torch.Tensor, h2: torch.Tensor, temperature: float = TEMPERATURE
) -> torch.Tensor:
    """Normalised-temperature cross-entropy over the 2N projections of N pairs.

    Each projection's positive is the other member of its pair; the other 2N - 2 are negatives.
    """
    n = h1.shape[0]
    h = F.normalize(torch.cat((h1, h2)), dim=1)
    logits = h @ h.T / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.cat((torch.arange(n, 2 * n), torch.arange(n))).to(h.device)
    return F.cross_entropy(logits, targets)
