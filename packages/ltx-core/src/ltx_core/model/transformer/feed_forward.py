import torch

from ltx_core.model.transformer.gelu_approx import GELUApprox


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Some sampler/attention paths promote activations to fp32 while the
        # native LTX checkpoints keep the FFN weights in bf16. Normalize at
        # the module boundary so Linear never receives a mismatched dtype.
        x = x.to(dtype=self.net[0].proj.weight.dtype)
        y = self.net[0](x)
        y = self.net[1](y)
        proj = self.net[2]
        with torch.autocast(device_type=y.device.type, enabled=False):
            y = torch.nn.functional.linear(
                y.float(), proj.weight.float(),
                proj.bias.float() if proj.bias is not None else None,
            )
        return y
