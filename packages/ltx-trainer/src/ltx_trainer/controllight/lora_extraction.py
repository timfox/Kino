"""LoRA extraction utilities for ControlLight hybrid training."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ltx_trainer.controllight.lora import StrengthScaledLoRA


def extract_lora_from_linear(
    linear: nn.Linear,
    rank: int = 64,
    alpha: int = 64,
    device: torch.device | str = "cpu",
) -> StrengthScaledLoRA:
    """Extract StrengthScaledLoRA from a trained Linear layer.

    The extracted LoRA approximates the trained weights W_trained as:
        W_trained ≈ W_base + s · (A @ B)

    where W_base is initialized as zeros (or can be kept as original initialization).

    Args:
        linear: Trained Linear layer to extract LoRA from
        rank: LoRA rank (default: 64)
        alpha: LoRA alpha parameter (default: 64)
        device: Device to place extracted LoRA on

    Returns:
        StrengthScaledLoRA instance with A and B initialized to approximate W_trained
    """
    in_f, out_f = linear.in_features, linear.out_features

    # Initialize base as zeros (learned adaptation starts from identity)
    base_init = nn.Linear(in_f, out_f, bias=linear.bias is not None, device=device)
    base_init.weight.data.zero_()
    if linear.bias is not None:
        base_init.bias.data.zero_()

    # Create LoRA with proper scaling
    lora = StrengthScaledLoRA(base_init, rank=rank, alpha=alpha)

    # Initialize A and B to approximate the trained weights
    # Use SVD decomposition: W_trained ≈ U Σ V^T, then A ≈ U Σ^(1/2), B ≈ Σ^(1/2) V^T
    with torch.no_grad():
        # Get the trained weights (we'll treat the linear layer's weights as "trained")
        # For extraction, we need to know what W_trained should be
        # Since we're extracting from a trained model, we'll use the current weights
        # as the target W_trained
        W_trained = linear.weight.data.to(device)

        # Center the weights (remove mean to better approximate low-rank structure)
        W_centered = W_trained - W_trained.mean(dim=0, keepdim=True)

        # SVD decomposition
        U, S, Vt = torch.linalg.svd(W_centered, full_matrices=False)

        # Take top rank components
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]

        # Initialize A and B: A = U_r @ diag(sqrt(S_r)), B = diag(sqrt(S_r)) @ Vt_r
        sqrt_S_r = torch.sqrt(S_r)
        lora.lora_a.weight.data = (U_r * sqrt_S_r.unsqueeze(0)).contiguous()
        lora.lora_b.weight.data = (sqrt_S_r.unsqueeze(-1) * Vt_r).contiguous()

    return lora


def extract_lora_from_module(
    module: nn.Module,
    target_modules: list[str],
    rank: int = 64,
    alpha: int = 64,
    device: torch.device | str = "cpu",
) -> dict[str, StrengthScaledLoRA]:
    """Extract LoRA from multiple linear layers in a module.

    Args:
        module: PyTorch module containing layers to extract LoRA from
        target_modules: List of module attribute names (e.g., ["to_q", "to_k", "to_v"])
        rank: LoRA rank
        alpha: LoRA alpha parameter
        device: Device for extracted LoRA

    Returns:
        Dictionary mapping module names to extracted StrengthScaledLoRA instances
    """
    lora_dict = {}
    for name in target_modules:
        attr = getattr(module, name, None)
        if isinstance(attr, nn.Linear):
            lora_dict[name] = extract_lora_from_linear(attr, rank=rank, alpha=alpha, device=device)
    return lora_dict


def apply_lora_to_module(
    module: nn.Module,
    lora_dict: dict[str, StrengthScaledLoRA],
    prefix: str = "",
) -> nn.Module:
    """Apply extracted LoRA to a module by replacing linear layers.

    Args:
        module: Module to apply LoRA to
        lora_dict: Dictionary of LoRA instances
        prefix: Module name prefix for recursive application

    Returns:
        Modified module with LoRA layers
    """
    for name, lora in lora_dict.items():
        full_name = f"{prefix}.{name}" if prefix else name
        if hasattr(module, name):
            # Replace the linear layer with the LoRA wrapper
            setattr(module, name, lora)
        else:
            # Try recursive application for nested modules
            for child_name, child in module.named_children():
                if child_name == name and isinstance(child, nn.Module):
                    apply_lora_to_module(child, {k: v for k, v in lora_dict.items() if k != name}, name)

    return module


def save_lora_adapters(lora_dict: dict[str, StrengthScaledLoRA], path: str) -> None:
    """Save LoRA adapters to a checkpoint file.

    Args:
        lora_dict: Dictionary of LoRA instances to save
        path: Path to save checkpoint
    """
    state_dict = {}
    for name, lora in lora_dict.items():
        # Save only LoRA parameters (not base)
        for param_name, param in lora.named_parameters():
            if param_name.startswith("lora_"):
                full_name = f"{name}.{param_name}"
                state_dict[full_name] = param.cpu()

    torch.save(state_dict, path)


def load_lora_adapters(lora_dict: dict[str, StrengthScaledLoRA], path: str) -> dict[str, StrengthScaledLoRA]:
    """Load LoRA adapters from a checkpoint file.

    Args:
        lora_dict: Dictionary of LoRA instances to load into
        path: Path to checkpoint file

    Returns:
        Updated lora_dict with loaded weights
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    for name, lora in lora_dict.items():
        lora_state = {}
        for key, value in state_dict.items():
            if key.startswith(f"{name}."):
                lora_state[key[len(f"{name}."):]] = value

        lora.load_state_dict(lora_state, strict=False)

    return lora_dict


def get_trainable_lora_params(lora_dict: dict[str, StrengthScaledLoRA]) -> list[nn.Parameter]:
    """Get all trainable parameters from LoRA adapters.

    Args:
        lora_dict: Dictionary of LoRA instances

    Returns:
        List of trainable parameters (A and B matrices only)
    """
    params = []
    for lora in lora_dict.values():
        for param in lora.parameters():
            if param.requires_grad:
                params.append(param)
    return params


def count_lora_params(lora_dict: dict[str, StrengthScaledLoRA]) -> int:
    """Count total trainable parameters in LoRA adapters.

    Args:
        lora_dict: Dictionary of LoRA instances

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in get_trainable_lora_params(lora_dict))


def lora_finetune_step(
    lora_dict: dict[str, StrengthScaledLoRA],
    batch: dict[str, Tensor],
    strength: float,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    device: torch.device | str = "cpu",
) -> float:
    """Perform one LoRA fine-tuning step.

    Args:
        lora_dict: Dictionary of LoRA instances
        batch: Training batch dictionary
        strength: LoRA strength parameter
        optimizer: Optimizer for LoRA parameters
        loss_fn: Loss function
        device: Device for computation

    Returns:
        Loss value
    """
    optimizer.zero_grad()

    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    # Forward pass with LoRA (strength modulation)
    # This would depend on how your model uses the LoRA layers
    # For now, we'll assume a simple interface
    output = lora_dict.get("output", None)  # Placeholder
    if output is None:
        raise ValueError("LoRA output not computed. Implement forward pass for your model.")

    loss = loss_fn(output, batch.get("target"))
    loss.backward()
    optimizer.step()

    return loss.item()