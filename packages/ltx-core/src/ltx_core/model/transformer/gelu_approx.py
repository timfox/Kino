import torch


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the input aligned with streamed/native checkpoint weights even
        # when an upstream attention kernel promotes activations to fp32.
        # PyTorch 2.11 + CUDA 13 can promote this streamed projection's
        # activation under the surrounding kernel context. Use a matched
        # fp32 projection explicitly; the caller casts the result back to
        # the native checkpoint dtype before the next Linear.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = torch.nn.functional.linear(
                x.float(), self.proj.weight.float(),
                self.proj.bias.float() if self.proj.bias is not None else None,
            )
        return torch.nn.functional.gelu(y, approximate="tanh")
