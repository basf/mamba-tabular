import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MambularConfig


### Heavily inspired and mostly taken from https://github.com/alxndrTL/mamba.py


class Mamba(nn.Module):
    """Mamba model composed of multiple MambaBlocks.

    Attributes:
        config (MambaConfig): Configuration object for the Mamba model.
        layers (nn.ModuleList): List of MambaBlocks constituting the model.
    """

    def __init__(self, config: MambularConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block composed of a MambaBlock and a normalization layer.

    Attributes:
        layers (MambaBlock): MambaBlock layers.
        norm (RMSNorm): Normalization layer.
    """

    def __init__(self, config: MambularConfig):
        super().__init__()

        self.layers = MambaBlock(config)
        self.norm = config.norm(config.d_model)

    def forward(self, x):
        output = self.layers(self.norm(x)) + x
        return output


class MambaBlock(nn.Module):
    """MambaBlock module containing the main computational components.

    Attributes:
        config (MambularConfig): Configuration object for the MambaBlock.
        in_proj (nn.Linear): Linear projection for input.
        conv1d (nn.Conv1d): 1D convolutional layer.
        x_proj (nn.Linear): Linear projection for input-dependent tensors.
        dt_proj (nn.Linear): Linear projection for dynamical time.
        A_log (nn.Parameter): Logarithmically stored A tensor.
        D (nn.Parameter): Tensor for D component.
        out_proj (nn.Linear): Linear projection for output.
    """

    def __init__(self, config: MambularConfig):
        super().__init__()

        self.config = config

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )

        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        _, L, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = F.silu(x)
        x = self.dropout(x)
        y = self.ssm(x)

        z = F.silu(z)
        z = self.dropout(z)

        output = y * z
        output = self.out_proj(output)

        return output

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)

        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        h = torch.zeros(
            x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device
        )
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y
