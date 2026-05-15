#!/usr/bin/env python3
"""Train the FiLM residual exposure head (``L_ev``) on precomputed LatentHDR-style stacks.

Requires each ``.pt`` under ``--latents-dir`` to contain at least:

- ``latents``: VAE latent tensor ``[C, F', H', W']`` (base / EV0 anchor),
- ``hdr_ldr_ev_stack``: ``[N, 3, F, H, W]`` γ-encoded synthetic LDRs (from ``process_videos`` with
  ``--hdr-ingest`` and ``--hdr-synth-bracket-ev``),
- ``hdr_ev_list``: length-``N`` list of EV offsets (float).

The frozen LTX VAE encoder is used to build per-EV **target** latents ``μ(x_e)`` (encoder means),
then ``L_ev = mean_e MSE(z_base + f_θ(z_base, φ(e)), μ(x_e))``.

This script does **not** implement joint ``L_diff + L_ev`` on the DiT; use the LatentHDR trainer GUI
exported YAML stub as a roadmap for wiring that into ``train.py`` later.

Example::

    uv run python scripts/train_latenthdr_exposure.py \\
        --latents-dir /data/.precomputed/latents \\
        --model-path /models/ltx2.safetensors \\
        --output /tmp/latenthdr_ev_head.pt \\
        --epochs 2 --device cuda
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import typer
from torch.utils.data import DataLoader, Dataset

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from process_videos import encode_video  # noqa: E402

from ltx_trainer.latenthdr import FiLMResidualExposureHead, exposure_latent_mse  # noqa: E402
from ltx_trainer.model_loader import load_video_vae_encoder  # noqa: E402

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


class BracketLatentFolderDataset(Dataset[dict[str, torch.Tensor | list[float]]]):
    """One sample per ``.pt`` file with ``latents`` + ``hdr_ldr_ev_stack`` + ``hdr_ev_list``."""

    def __init__(self, root: Path, max_samples: int | None = None) -> None:
        self.paths = sorted(root.rglob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files under {root}")
        if max_samples is not None:
            self.paths = self.paths[: max_samples]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[float]]:
        path = self.paths[index]
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in {path}, got {type(data)}")
        for key in ("latents", "hdr_ldr_ev_stack", "hdr_ev_list"):
            if key not in data:
                raise KeyError(f"{path} missing {key!r} (need HDR preprocess with --hdr-synth-bracket-ev)")
        lat = data["latents"]
        stack = data["hdr_ldr_ev_stack"]
        evs = data["hdr_ev_list"]
        if not isinstance(evs, list) or len(evs) != stack.shape[0]:
            raise ValueError(f"{path}: hdr_ev_list length must match hdr_ldr_ev_stack dim 0")
        if stack.shape[1] != 3:
            raise ValueError(f"{path}: hdr_ldr_ev_stack must have 3 RGB channels, got {stack.shape}")
        return {
            "latents": lat.float(),
            "stack": stack.float(),
            "hdr_ev_list": [float(x) for x in evs],
        }


def _collate_single(batch: list[dict[str, torch.Tensor | list[float]]]) -> dict[str, torch.Tensor | list[float]]:
    if len(batch) != 1:
        raise ValueError("Use batch_size=1 with this collate function")
    return batch[0]


def _encode_ev_stack_targets(
    vae: torch.nn.Module,
    stack_bf16: torch.Tensor,
    *,
    latent_dtype: torch.dtype,
    use_tiling: bool,
) -> torch.Tensor:
    """Encode ``[N,3,F,H,W]`` γ-LDRs to latent means ``[N,C,F',H',W']``."""
    device = stack_bf16.device
    n = stack_bf16.shape[0]
    outs: list[torch.Tensor] = []
    for i in range(n):
        vid = stack_bf16[i : i + 1]
        inp = (vid.clamp(0.0, 1.0) - 0.5) / 0.5
        with torch.no_grad():
            enc = encode_video(vae, inp, dtype=latent_dtype, use_tiling=use_tiling)
        z = enc["latents"][0]
        outs.append(z.float())
    return torch.stack(outs, dim=0)


@app.command()
def main(
    latents_dir: str = typer.Argument(..., help="Directory tree of .pt files (e.g. .precomputed/latents)"),
    model_path: str = typer.Argument(..., help="LTX-2 .safetensors (frozen VAE encoder)"),
    output: str = typer.Argument(..., help="Output path for trained head weights (.pt)"),
    epochs: int = typer.Option(1, help="Training epochs"),
    lr: float = typer.Option(5e-5, help="Adam learning rate"),
    device: str = typer.Option("cuda", help="cuda or cpu"),
    max_samples: int | None = typer.Option(None, help="Limit number of .pt files (smoke / debug)"),
    vae_tiling: bool = typer.Option(False, help="Enable VAE spatial tiling when encoding targets"),
    latent_channels: int = typer.Option(128, help="Latent channel width (must match checkpoint)"),
    cond_dim: int = typer.Option(128, help="EV conditioning MLP output width"),
    num_layers: int = typer.Option(4, help="FiLM Conv3d blocks"),
    residual_scale: float = typer.Option(0.05, help="Initial residual multiplier on Δz"),
    target_latent_dtype: str = typer.Option("float32", help="dtype for encoded targets: float32 or bfloat16"),
) -> None:
    root = Path(latents_dir)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if target_latent_dtype not in dtype_map:
        raise typer.BadParameter(f"target_latent_dtype must be one of {list(dtype_map)}")
    latent_dtype = dtype_map[target_latent_dtype]

    torch_device = torch.device(device)
    ds = BracketLatentFolderDataset(root, max_samples=max_samples)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=_collate_single)

    vae = load_video_vae_encoder(model_path, device=torch_device, dtype=torch.bfloat16)
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    head = FiLMResidualExposureHead(
        latent_channels=latent_channels,
        cond_dim=cond_dim,
        num_layers=num_layers,
        residual_scale=residual_scale,
    ).to(torch_device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0.0
        count = 0
        for batch in dl:
            z_base = batch["latents"].to(torch_device)
            stack = batch["stack"].to(torch_device)
            ev_list_t = batch["hdr_ev_list"]
            if not isinstance(ev_list_t, list):
                raise TypeError("hdr_ev_list must be a list of floats")

            zb = z_base if z_base.ndim == 5 else z_base.unsqueeze(0)

            stack_bf16 = stack.to(device=torch_device, dtype=torch.bfloat16)
            targets = _encode_ev_stack_targets(
                vae, stack_bf16, latent_dtype=latent_dtype, use_tiling=vae_tiling
            ).to(torch_device)

            loss_acc = torch.zeros((), device=torch_device)
            for i, ev in enumerate(ev_list_t):
                ev_t = torch.full((1,), ev, device=torch_device, dtype=torch.float32)
                pred = head(zb, ev_t)[0]
                loss_acc = loss_acc + exposure_latent_mse(pred, targets[i])
            loss = loss_acc / len(ev_list_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu())
            count += 1

        typer.echo(f"epoch {epoch + 1}/{epochs} mean L_ev: {total / max(count, 1):.6f}")

    torch.save(
        {
            "exposure_head": head.state_dict(),
            "config": {
                "latent_channels": latent_channels,
                "cond_dim": cond_dim,
                "num_layers": num_layers,
                "residual_scale": residual_scale,
                "model_path": model_path,
                "latents_dir": str(root),
            },
        },
        out_path,
    )
    typer.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
