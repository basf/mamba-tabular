from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultMLPConfig:
    """
    Configuration class for the default Multi-Layer Perceptron (MLP) model with predefined hyperparameters.

    Optimizer Parameters
    --------------------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 regularization) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.

    MLP Architecture Parameters
    ---------------------------
    layer_sizes : list, default=(256, 128, 32)
        Sizes of the layers in the MLP.
    activation : callable, default=nn.SELU()
        Activation function for the MLP layers.
    skip_layers : bool, default=False
        Whether to skip layers in the MLP.
    dropout : float, default=0.5
        Dropout rate for regularization.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the MLP.
    skip_connections : bool, default=False
        Whether to use skip connections in the MLP.
    batch_norm : bool, default=False
        Whether to use batch normalization in the MLP layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the MLP layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.

    Embedding Parameters
    ---------------------
    use_embeddings : bool, default=False
        Whether to use embedding layers for all features.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', 'plr', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    d_model : int, default=32
        Dimensionality of the embeddings.
    plr_lite : bool, default=False
        Whether to use a lightweight version of Piecewise Linear Regression (PLR).
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    layer_sizes: list = (256, 128, 32)
    activation: callable = nn.ReLU()
    skip_layers: bool = False
    dropout: float = 0.2
    use_glu: bool = False
    skip_connections: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    layer_norm_eps: float = 1e-05
    use_embeddings: bool = False
    embedding_activation: callable = nn.Identity()
    embedding_type: str = "linear"
    embedding_bias: bool = False
    layer_norm_after_embedding: bool = False
    d_model: int = 32
    plr_lite: bool = False
