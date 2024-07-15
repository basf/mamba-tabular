from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultRotaryFTTransformerConfig:
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    attn_dropout: float = 0.3
    ff_dropout: float = 0.3
    norm: str = "LayerNorm"
    activation: callable = nn.SELU()
    num_embedding_activation: callable = nn.Identity()
    head_layer_sizes: list = (128, 64, 32)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: callable = nn.SELU()
    head_use_batch_norm: bool = False
    layer_norm_after_embedding: bool = False
    pooling_method: str = "sum"
    norm_first: bool = False
    bias: bool = True
    transformer_activation: callable = nn.ReLU()
    layer_norm_eps: float = 1e-07
    transformer_dim_feedforward: int = 512