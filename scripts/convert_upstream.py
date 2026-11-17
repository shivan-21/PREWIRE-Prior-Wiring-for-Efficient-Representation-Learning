"""Convert an upstream LPL (Lightning) checkpoint into our ``encoder.pt`` format.

    python scripts/convert_upstream.py <upstream.ckpt> results_and_figs/checkpoints/zenke_reference

Upstream wraps the same ``VGG11Encoder`` as ``network.encoder``; stripping that prefix gives a
state dict that ``src.models.vgg.load_encoder`` accepts, so the evaluation scripts can score
upstream's own network with exactly the same protocol as ours.
"""

import argparse
from pathlib import Path

import torch

from src.models.vgg import vgg11

PREFIX = "network.encoder."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="upstream Lightning .ckpt file")
    parser.add_argument("out_dir", help="output directory; encoder.pt is written inside")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    encoder_state = {k[len(PREFIX) :]: v for k, v in state.items() if k.startswith(PREFIX)}
    vgg11(layer_local=True, init=False).load_state_dict(encoder_state)  # fails if keys differ

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": encoder_state, "source": str(args.checkpoint)}, out / "encoder.pt")
    print(f"wrote {out / 'encoder.pt'} ({len(encoder_state)} tensors)")


if __name__ == "__main__":
    main()
