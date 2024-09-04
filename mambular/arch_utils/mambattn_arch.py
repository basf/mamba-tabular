import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization_layers import RMSNorm
from .mamba_arch import ResidualBlock


class MambAttn(nn.Module):
    """Mamba model composed of alternating MambaBlocks and Attention layers.

    Attributes:
        config (MambaConfig): Configuration object for the Mamba model.
        layers (nn.ModuleList): List of alternating ResidualBlock (Mamba layers) and attention layers constituting the model.
    """

    def __init__(
        self,
        d_model=32,
        n_layers=8,
        n_attention_layers=1,  # Introduce attention layer count
        n_mamba_per_attention=1,  # Ratio of Mamba layers to attention layers
        n_heads=4,  # Number of attention heads
        expand_factor=2,
        bias=False,
        d_conv=8,
        conv_bias=True,
        dropout=0.0,
        attn_dropout=0.1,
        dt_rank="auto",
        d_state=16,
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        last_layer="attn",  # Define the desired last layer type
        dt_min=1e-03,
        dt_init_floor=1e-04,
        norm=RMSNorm,
        activation=F.silu,
        bidirectional=False,
        use_learnable_interaction=False,
        layer_norm_eps=1e-05,
        AD_weight_decay=False,
        BC_layer_norm=True,
    ):
        super().__init__()

        # Define Mamba and Attention layers alternation
        self.layers = nn.ModuleList()

        total_blocks = n_layers + n_attention_layers  # Total blocks to be created
        attention_count = 0

        for i in range(total_blocks):
            if (i + 1) % (
                n_mamba_per_attention + 1
            ) == 0:  # Insert attention layer after N Mamba layers
                self.layers.append(
                    nn.MultiheadAttention(
                        embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout
                    )
                )
                attention_count += 1
            else:
                self.layers.append(
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
                        layer_norm_eps,
                        AD_weight_decay,
                        BC_layer_norm,
                    )
                )

        # Check the type of the last layer and append the desired one if necessary
        if last_layer == "attn":
            if not isinstance(self.layers[-1], nn.MultiheadAttention):
                self.layers.append(
                    nn.MultiheadAttention(
                        embed_dim=d_model, num_heads=n_heads, dropout=dropout
                    )
                )
        else:
            if not isinstance(self.layers[-1], ResidualBlock):
                self.layers.append(
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
                        layer_norm_eps,
                        AD_weight_decay,
                        BC_layer_norm,
                    )
                )

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                # If it's an attention layer, handle input shape (seq_len, batch, embed_dim)
                x = x.transpose(
                    0, 1
                )  # Switch to (seq_len, batch, embed_dim) for attention
                x, _ = layer(x, x, x)
                x = x.transpose(0, 1)  # Switch back to (batch, seq_len, embed_dim)
            else:
                # Otherwise, pass through Mamba block
                x = layer(x)

        return x
