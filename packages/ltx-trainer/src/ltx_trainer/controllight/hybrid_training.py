"""Hybrid training pipeline: Phase 1 (full UNet) → Phase 2 (LoRA extraction) → Phase 3 (LoRA inference)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ltx_trainer.controllight.config import ControlLightConfig
from ltx_trainer.controllight.lora import StrengthScaledLoRA
from ltx_trainer.controllight.lora_extraction import (
    extract_lora_from_linear,
    extract_lora_from_module,
    save_lora_adapters,
    load_lora_adapters,
    count_lora_params,
)
from ltx_trainer.controllight.pipeline import controllight_training_loss, prepare_training_batch


class HybridControlLightTrainer:
    """Hybrid trainer supporting full UNet training and LoRA extraction.

    Phase 1: Train full UNet on Light100K pairs
    Phase 2: Extract LoRA adapters from trained UNet weights
    Phase 3: Use LoRA for inference with strength control
    """

    def __init__(
        self,
        unet: nn.Module,
        cfg: ControlLightConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or ControlLightConfig()
        self.device = torch.device(device)
        self.unet = unet.to(self.device)

        # Phase 1: Full UNet training
        self.unet.train()

        # Phase 2: LoRA extraction (initialized, not yet trained)
        self.lora_dict: dict[str, StrengthScaledLoRA] = {}
        self.lora_extracted = False

        # Phase 3: LoRA inference mode
        self.lora_inference_mode = False

    def phase1_train_full_unet(
        self,
        i0: Tensor,
        i1: Tensor,
        strength: float,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Phase 1: Train full UNet on a single batch.

        Args:
            i0: Low-light input tensor
            i1: High-light target tensor
            strength: Enhancement strength (0.2, 0.4, 0.6, 0.8, 1.0)
            optimizer: Optimizer for UNet parameters

        Returns:
            Dictionary with loss values
        """
        optimizer.zero_grad()

        # Prepare training batch
        target, group = prepare_training_batch(i0, i1, strength, cfg=self.cfg)

        # Forward pass through full UNet
        z0 = self._encode(i0)
        z1 = self._encode(target)
        v_pred = self.unet(z0, t=torch.tensor([0.5], device=self.device))

        # Compute loss
        loss = controllight_training_loss(v_pred, z0, z1, i0, torch.ones_like(i0[:, 0, 0, 0]), cfg=self.cfg)

        # Backward pass
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    def phase2_extract_lora(self, target_modules: list[str] | None = None) -> dict[str, StrengthScaledLoRA]:
        """Phase 2: Extract LoRA adapters from trained UNet.

        Args:
            target_modules: List of module names to extract LoRA from
                          (default: ["to_q", "to_k", "to_v", "to_out.0"] for attention layers)

        Returns:
            Dictionary of extracted StrengthScaledLoRA instances
        """
        if target_modules is None:
            # Default: extract from attention layers
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

        # Extract LoRA from each target module
        self.lora_dict = extract_lora_from_module(
            self.unet,
            target_modules=target_modules,
            rank=self.cfg.lora_rank,
            alpha=self.cfg.lora_alpha,
            device=self.device,
        )

        self.lora_extracted = True

        # Print parameter counts
        total_lora_params = count_lora_params(self.lora_dict)
        total_unet_params = sum(p.numel() for p in self.unet.parameters())

        print(f"Phase 2: Extracted LoRA adapters")
        print(f"  Total UNet parameters: {total_unet_params:,}")
        print(f"  Total LoRA parameters: {total_lora_params:,}")
        print(f"  Parameter reduction: {(1 - total_lora_params / total_unet_params) * 100:.2f}%")

        return self.lora_dict

    def phase3_lora_inference(
        self,
        x: Tensor,
        strength: float = 1.0,
        use_lora: bool = True,
    ) -> Tensor:
        """Phase 3: Inference with LoRA strength control.

        Args:
            x: Input tensor
            strength: LoRA strength (0.0 to 1.0)
            use_lora: Whether to use LoRA or full UNet

        Returns:
            Output tensor
        """
        if use_lora and self.lora_extracted:
            # Use LoRA with strength modulation
            return self._lora_forward(x, strength)
        else:
            # Use full UNet
            return self._unet_forward(x)

    def _unet_forward(self, x: Tensor) -> Tensor:
        """Forward pass through full UNet."""
        return self.unet(x, t=torch.tensor([0.5], device=self.device))

    def _lora_forward(self, x: Tensor, strength: float) -> Tensor:
        """Forward pass through LoRA-adapted UNet."""
        # This would require modifying the UNet to use LoRA layers
        # For now, we'll use the full UNet as a placeholder
        # In practice, you'd replace the attention layers with LoRA wrappers
        return self._unet_forward(x)

    def _encode(self, x: Tensor) -> Tensor:
        """Encode input to latent space (placeholder)."""
        # This would be your VAE encoder
        return x  # Placeholder

    def save_lora_checkpoint(self, path: str) -> None:
        """Save LoRA adapters to checkpoint."""
        if not self.lora_extracted:
            raise ValueError("LoRA not extracted yet. Call phase2_extract_lora() first.")
        save_lora_adapters(self.lora_dict, path)
        print(f"Saved LoRA checkpoint to {path}")

    def load_lora_checkpoint(self, path: str) -> None:
        """Load LoRA adapters from checkpoint."""
        self.lora_dict = load_lora_adapters(self.lora_dict, path)
        self.lora_extracted = True
        print(f"Loaded LoRA checkpoint from {path}")


def create_hybrid_trainer_from_checkpoint(
    unet: nn.Module,
    checkpoint_path: str,
    cfg: ControlLightConfig | None = None,
    device: torch.device | str = "cpu",
) -> HybridControlLightTrainer:
    """Create hybrid trainer from existing UNet checkpoint.

    Args:
        unet: UNet model instance
        checkpoint_path: Path to UNet checkpoint
        cfg: Configuration
        device: Device

    Returns:
        HybridControlLightTrainer with loaded UNet weights
    """
    trainer = HybridControlLightTrainer(unet, cfg=cfg, device=device)

    # Load UNet weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trainer.unet.load_state_dict(state_dict)
    print(f"Loaded UNet checkpoint from {checkpoint_path}")

    return trainer


def example_hybrid_training_workflow() -> None:
    """Example workflow for hybrid training."""
    print("=" * 60)
    print("ControlLight Hybrid Training Workflow")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ControlLightConfig()

    # Placeholder UNet (replace with actual FLUX.2-klein UNet)
    unet = nn.Identity()  # Placeholder

    # Create trainer
    trainer = HybridControlLightTrainer(unet, cfg=cfg, device=device)

    # Phase 1: Train full UNet
    print("\nPhase 1: Training full UNet")
    print("-" * 40)

    # Dummy data
    i0 = torch.randn(1, 3, 256, 256, device=device)  # Low-light input
    i1 = torch.randn(1, 3, 256, 256, device=device)  # High-light target

    # Optimizer for full UNet
    optimizer_unet = torch.optim.AdamW(trainer.unet.parameters(), lr=cfg.learning_rate)

    # Training loop (simplified)
    for step in range(10):  # Replace with actual training loop
        loss_info = trainer.phase1_train_full_unet(i0, i1, strength=0.6, optimizer=optimizer_unet)
        if step % 2 == 0:
            print(f"  Step {step}: loss = {loss_info['loss']:.4f}")

    print("\nPhase 1 complete: Full UNet trained")

    # Phase 2: Extract LoRA
    print("\nPhase 2: Extracting LoRA adapters")
    print("-" * 40)

    lora_dict = trainer.phase2_extract_lora()

    print("\nPhase 2 complete: LoRA adapters extracted")

    # Phase 3: LoRA inference
    print("\nPhase 3: LoRA inference with strength control")
    print("-" * 40)

    # Test different strengths
    for strength in [0.25, 0.5, 0.75, 1.0]:
        output = trainer.phase3_lora_inference(i0, strength=strength, use_lora=True)
        print(f"  Strength {strength}: output shape = {output.shape}")

    print("\nPhase 3 complete: LoRA inference demonstrated")

    # Save LoRA checkpoint
    print("\nSaving LoRA checkpoint...")
    trainer.save_lora_checkpoint("controllight_lora_checkpoint.pt")

    print("\n" + "=" * 60)
    print("Hybrid training workflow complete!")
    print("=" * 60)


if __name__ == "__main__":
    example_hybrid_training_workflow()