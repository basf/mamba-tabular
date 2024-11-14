from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultResNetConfig:
    """
    Configuration class for the default ResNet model with predefined hyperparameters.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    layer_sizes : list, default=(128, 128, 32)
        Sizes of the layers in the ResNet.
    activation : callable, default=nn.SELU()
        Activation function for the ResNet layers.
    skip_layers : bool, default=False
        Whether to skip layers in the ResNet.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the ResNet.
    skip_connections : bool, default=True
        Whether to use skip connections in the ResNet.
    batch_norm : bool, default=True
        Whether to use batch normalization in the ResNet layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the ResNet layers.
    num_blocks : int, default=3
        Number of residual blocks in the ResNet.
    use_embeddings : bool, default=False
        Whether to use embedding layers for all features.
    embedding_activation : callable, default=nn.Identity()
        Activation function for  embeddings.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    d_model : int, default=32
        Dimensionality of the embeddings.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    layer_sizes: list = (256, 128, 32)
    activation: callable = nn.SELU()
    skip_layers: bool = False
    dropout: float = 0.5
    norm: str = None
    use_glu: bool = False
    skip_connections: bool = True
    batch_norm: bool = True
    layer_norm: bool = False
    layer_norm_eps: float = 1e-05
    num_blocks: int = 3

    # embedding params
    use_embeddings: bool = True
    embedding_type: float = "plr"
    plr_lite: bool = False
    average_embeddings: bool = True
    embedding_activation: callable = nn.Identity()
    layer_norm_after_embedding: bool = False
    d_model: int = 64
