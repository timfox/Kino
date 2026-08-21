"""Boolean convolution and indicator activation (Sec. 1.2.2, Eq. 6–7)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def indicator(x: Tensor) -> Tensor:
    """I(x > 0) comparison-based ReLU replacement."""
    return (x > 0).to(x.dtype)


class BooleanConv2d(nn.Module):
    """
    Quantized conv: Y = I(S_P - S_N > 0) where P/N index ±1 weights.

    Input and effective weights are treated as binary {0,1} for AND semantics.
    """

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        w = weight.detach()
        self.register_buffer("w_pos", (w == 1).to(torch.float32))
        self.register_buffer("w_neg", (w == -1).to(torch.float32))
        self.num_nonzero = int((w != 0).sum().item())

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C_in, H, W); w_pos/w_neg: (C_out, C_in, kH, kW)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        b, _, h, w = x.shape
        out_c = self.w_pos.shape[0]
        k_h, k_w = self.w_pos.shape[2], self.w_pos.shape[3]
        pad_h, pad_w = k_h // 2, k_w // 2
        patches = torch.nn.functional.unfold(
            x, kernel_size=(k_h, k_w), padding=(pad_h, pad_w)
        )  # (B, C_in*kH*kW, L)
        pos_flat = self.w_pos.reshape(out_c, -1)  # (C_out, C_in*kH*kW)
        neg_flat = self.w_neg.reshape(out_c, -1)
        sp = torch.einsum("oc,bpl->bol", pos_flat, patches)
        sn = torch.einsum("oc,bpl->bol", neg_flat, patches)
        y = indicator(sp - sn)
        oh = h
        ow = w
        return y.view(b, out_c, oh, ow)

    def operation_counts(self, output_elements: int) -> dict[str, int]:
        nz = self.num_nonzero
        out_c = self.w_pos.shape[0]
        return {
            "boolean": nz * output_elements,
            "additions": max(0, nz - out_c) * output_elements,
            "indicators": out_c * output_elements,
        }


class BooleanLinear(nn.Module):
    """Fully connected layer as Boolean sum comparison (prediction head)."""

    def __init__(self, weight: Tensor, bias: Tensor | None = None) -> None:
        super().__init__()
        w = weight.detach()
        self.register_buffer("w_pos", (w == 1).to(torch.float32))
        self.register_buffer("w_neg", (w == -1).to(torch.float32))
        self.num_nonzero = int((w != 0).sum().item())
        self.out_features = w.shape[0]
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        sp = torch.matmul(x, self.w_pos.t())
        sn = torch.matmul(x, self.w_neg.t())
        return indicator(sp - sn)

    def operation_counts(self, batch: int = 1) -> dict[str, int]:
        nz = self.num_nonzero
        out = self.out_features
        elems = batch * out
        return {
            "boolean": nz * batch,
            "additions": max(0, nz - out) * batch,
            "indicators": elems,
        }
