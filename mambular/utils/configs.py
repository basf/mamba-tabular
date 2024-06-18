from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultMambularConfig:
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    d_model: int = 64
    n_layers: int = 8
    expand_factor: int = 2
    bias: bool = False
    d_conv: int = 16
    conv_bias: bool = True
    dropout: float = 0.05
    dt_rank: str = "auto"
    d_state: int = 32
    dt_scale: float = 1.0
    dt_init: str = "random"
    dt_max: float = 0.1
    dt_min: float = 1e-04
    dt_init_floor: float = 1e-04
    norm: str = "RMSNorm"
    activation: callable = nn.SELU()
    num_embedding_activation: callable = nn.Identity()
    head_layer_sizes: list = (128, 64, 32)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: callable = nn.SELU()
    head_use_batch_norm: bool = False
    layer_norm_after_embedding: bool = False
    pooling_method: str = "avg"


@dataclass
class DefaultFTTransformerConfig:
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    d_model: int = 64
    n_layers: int = 8
    n_heads: int = 4
    attn_dropout: float = 0.3
    ff_dropout: float = 0.3
    norm: str = "RMSNorm"
    activation: callable = nn.SELU()
    num_embedding_activation: callable = nn.Identity()
    head_layer_sizes: list = (128, 64, 32)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: callable = nn.SELU()
    head_use_batch_norm: bool = False
    layer_norm_after_embedding: bool = False
    pooling_method: str = "cls"
    norm_first: bool = False
    bias: bool = True
    transformer_activation: callable = nn.SELU()
    layer_norm_eps: float = 1e-05
    transformer_dim_feedforward: int = 2048


@dataclass
class DefaultMLPConfig:
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    layer_sizes: list = (128, 128, 32)
    activation: callable = nn.SELU()
    skip_layers: bool = False
    dropout: float = 0.5
    norm: str = None
    use_glu: bool = False


@dataclass
class DefaultResNetConfig:
    hidden_dims: list = (128, 64, 32)
    activation: callable = nn.SELU()
    dropout: float = 0.2
    batch_norm: bool = True
    norm: str = "BatchNorm"
    num_blocks: int = 3
    lr: float = 0.001
    lr_patience: int = 10
    weight_decay: float = 1e-4
    lr_factor: float = 0.5
