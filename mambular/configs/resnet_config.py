from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultResNetConfig:
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
    skip_connections: bool = True
    batch_norm: bool = True
    layer_norm: bool = False
    num_blocks: int = 3
