import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)


### Heavily inspired and mostly taken from https://github.com/alxndrTL/mamba.py


class Mamba(nn.Module):
    """Mamba model composed of multiple MambaBlocks.

    Attributes:
        config (MambaConfig): Configuration object for the Mamba model.
        layers (nn.ModuleList): List of MambaBlocks constituting the model.
    """

    def __init__(
        self,
        d_model=32,
        n_layers=8,
        expand_factor=2,
        bias=False,
        d_conv=8,
        conv_bias=True,
        dropout=0.01,
        dt_rank="auto",
        d_state=16,
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        norm=RMSNorm,
        activation=F.silu,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_model,
                    expand_factor,
                    bias,
                    d_conv,
                    conv_bias,
                    dropout,
                    dt_rank,
                    d_state,
                    dt_scale,
                    dt_init,
                    dt_max,
                    dt_min,
                    dt_init_floor,
                    norm,
                    activation,
                )
                for _ in range(n_layers)
            ]
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

    def __init__(
        self,
        d_model=32,
        expand_factor=2,
        bias=False,
        d_conv=16,
        conv_bias=True,
        dropout=0.01,
        dt_rank="auto",
        d_state=32,
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        norm=RMSNorm,
        activation=F.silu,
    ):
        super().__init__()

        VALID_NORMALIZATION_LAYERS = {
            "RMSNorm": RMSNorm,
            "LayerNorm": LayerNorm,
            "LearnableLayerScaling": LearnableLayerScaling,
            "BatchNorm": BatchNorm,
            "InstanceNorm": InstanceNorm,
            "GroupNorm": GroupNorm,
        }

        # Check if the provided normalization layer is valid
        if isinstance(norm, type) and norm.__name__ not in VALID_NORMALIZATION_LAYERS:
            raise ValueError(
                f"Invalid normalization layer: {norm.__name__}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )
        elif isinstance(norm, str) and norm not in self.VALID_NORMALIZATION_LAYERS:
            raise ValueError(
                f"Invalid normalization layer: {norm}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )

        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)

        self.layers = MambaBlock(
            d_model=d_model,
            expand_factor=expand_factor,
            bias=bias,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            dt_rank=dt_rank,
            d_state=d_state,
            dt_scale=dt_scale,
            dt_init=dt_init,
            dt_max=dt_max,
            dt_min=dt_min,
            dt_init_floor=dt_init_floor,
            activation=activation,
        )
        self.norm = norm(d_model)

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

    def __init__(
        self,
        d_model=32,
        expand_factor=2,
        bias=False,
        d_conv=16,
        conv_bias=True,
        dropout=0.01,
        dt_rank="auto",
        d_state=32,
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        activation=F.silu,
    ):
        super().__init__()
        self.d_inner = d_model * expand_factor

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)

        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dt_rank = dt_rank
        self.d_state = d_state

    def forward(self, x):
        _, L, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = self.activation(x)
        x = self.dropout(x)
        y = self.ssm(x)

        z = self.activation(z)
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
            [self.dt_rank, self.d_state, self.d_state],
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

        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=deltaA.device)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y
