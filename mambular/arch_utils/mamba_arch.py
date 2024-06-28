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
        bidirectional=False,
        use_learnable_interaction=False,
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
                    bidirectional,
                    use_learnable_interaction,
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
        bidirectional=False,
        use_learnable_interaction=False,
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
            bidirectional=bidirectional,
            use_learnable_interaction=use_learnable_interaction,
        )
        self.norm = norm(d_model)

    def forward(self, x):
        output = self.layers(self.norm(x)) + x
        return output


class MambaBlock(nn.Module):
    """MambaBlock module containing the main computational components.

    Attributes:
        in_proj (nn.Linear): Linear projection for input.
        conv1d (nn.Conv1d): 1D convolutional layer.
        x_proj (nn.Linear): Linear projection for input-dependent tensors.
        dt_proj (nn.Linear): Linear projection for dynamical time.
        A_log (nn.Parameter): Logarithmically stored A tensor.
        D (nn.Parameter): Tensor for D component.
        out_proj (nn.Linear): Linear projection for output.
        learnable_interaction (LearnableFeatureInteraction): Learnable feature interaction layer.
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
        bidirectional=False,
        use_learnable_interaction=False,
    ):
        super().__init__()
        self.d_inner = d_model * expand_factor
        self.bidirectional = bidirectional
        self.use_learnable_interaction = use_learnable_interaction

        self.in_proj_fwd = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        if self.bidirectional:
            self.in_proj_bwd = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        self.conv1d_fwd = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        if self.bidirectional:
            self.conv1d_bwd = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=self.d_inner,
                padding=d_conv - 1,
            )

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        if self.use_learnable_interaction:
            self.learnable_interaction = LearnableFeatureInteraction(self.d_inner)

        self.x_proj_fwd = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        if self.bidirectional:
            self.x_proj_bwd = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)

        self.dt_proj_fwd = nn.Linear(dt_rank, self.d_inner, bias=True)
        if self.bidirectional:
            self.dt_proj_bwd = nn.Linear(dt_rank, self.d_inner, bias=True)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_fwd.weight, dt_init_std)
            if self.bidirectional:
                nn.init.constant_(self.dt_proj_bwd.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_fwd.weight, -dt_init_std, dt_init_std)
            if self.bidirectional:
                nn.init.uniform_(self.dt_proj_bwd.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt_fwd = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt_fwd = dt_fwd + torch.log(-torch.expm1(-dt_fwd))
        with torch.no_grad():
            self.dt_proj_fwd.bias.copy_(inv_dt_fwd)

        if self.bidirectional:
            dt_bwd = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt_bwd = dt_bwd + torch.log(-torch.expm1(-dt_bwd))
            with torch.no_grad():
                self.dt_proj_bwd.bias.copy_(inv_dt_bwd)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_fwd = nn.Parameter(torch.log(A))
        if self.bidirectional:
            self.A_log_bwd = nn.Parameter(torch.log(A))

        self.D_fwd = nn.Parameter(torch.ones(self.d_inner))
        if self.bidirectional:
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dt_rank = dt_rank
        self.d_state = d_state

    def forward(self, x):
        _, L, _ = x.shape

        xz_fwd = self.in_proj_fwd(x)
        x_fwd, z_fwd = xz_fwd.chunk(2, dim=-1)

        x_fwd = x_fwd.transpose(1, 2)
        x_fwd = self.conv1d_fwd(x_fwd)[:, :, :L]
        x_fwd = x_fwd.transpose(1, 2)

        if self.bidirectional:
            xz_bwd = self.in_proj_bwd(x)
            x_bwd, z_bwd = xz_bwd.chunk(2, dim=-1)

            x_bwd = x_bwd.transpose(1, 2)
            x_bwd = self.conv1d_bwd(x_bwd)[:, :, :L]
            x_bwd = x_bwd.transpose(1, 2)

        if self.use_learnable_interaction:
            x_fwd = self.learnable_interaction(x_fwd)
            if self.bidirectional:
                x_bwd = self.learnable_interaction(x_bwd)

        x_fwd = self.activation(x_fwd)
        x_fwd = self.dropout(x_fwd)
        y_fwd = self.ssm(x_fwd, forward=True)

        if self.bidirectional:
            x_bwd = self.activation(x_bwd)
            x_bwd = self.dropout(x_bwd)
            y_bwd = self.ssm(torch.flip(x_bwd, [1]), forward=False)
            y = y_fwd + torch.flip(y_bwd, [1])
        else:
            y = y_fwd

        z_fwd = self.activation(z_fwd)
        z_fwd = self.dropout(z_fwd)

        output = y * z_fwd
        output = self.out_proj(output)

        return output

    def ssm(self, x, forward=True):
        if forward:
            A = -torch.exp(self.A_log_fwd.float())
            D = self.D_fwd.float()
            deltaBC = self.x_proj_fwd(x)
            delta, B, C = torch.split(
                deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            delta = F.softplus(self.dt_proj_fwd(delta))
        else:
            A = -torch.exp(self.A_log_bwd.float())
            D = self.D_bwd.float()
            deltaBC = self.x_proj_bwd(x)
            delta, B, C = torch.split(
                deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            delta = F.softplus(self.dt_proj_bwd(delta))

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


class LearnableFeatureInteraction(nn.Module):
    def __init__(self, n_vars):
        super().__init__()
        self.interaction_weights = nn.Parameter(torch.Tensor(n_vars, n_vars))
        nn.init.xavier_uniform_(self.interaction_weights)

    def forward(self, x):
        batch_size, n_vars, d_model = x.size()
        interactions = torch.matmul(x, self.interaction_weights)
        return interactions.view(batch_size, n_vars, d_model)
