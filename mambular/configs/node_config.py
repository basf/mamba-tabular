from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultNODEConfig:
    """
    Configuration class for the Neural Oblivious Decision Ensemble (NODE) model.

    Optimizer Parameters
    --------------------
    lr : float, default=1e-03
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs without improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 regularization penalty) applied by the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate is reduced when there is no improvement.

    Model Architecture Parameters
    -----------------------------
    num_layers : int, default=4
        Number of dense layers in the model.
    layer_dim : int, default=128
        Dimensionality of each dense layer.
    tree_dim : int, default=1
        Dimensionality of the output from each tree leaf.
    depth : int, default=6
        Depth of each decision tree in the ensemble.
    norm : str, default=None
        Type of normalization to use in the model.

    Embedding Parameters
    ---------------------
    use_embeddings : bool, default=False
        Whether to use embedding layers for categorical features.
    embedding_activation : callable, default=nn.Identity()
        Activation function to apply to embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding layers.
    d_model : int, default=32
        Dimensionality of the embedding space.

    Head Parameters
    ---------------
    head_layer_sizes : list, default=()
        Sizes of the layers in the model's head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    """

    lr: float = 1e-03
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    norm: str = None
    use_embeddings: bool = False
    embedding_activation: callable = nn.Identity()
    embedding_tpye: str = "linear"
    embedding_bias: bool = False
    layer_norm_after_embedding: bool = False
    d_model: int = 32
    num_layers: int = 4
    layer_dim: int = 128
    tree_dim: int = 1
    depth: int = 6
    head_layer_sizes: list = ()
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: callable = nn.SELU()
    head_use_batch_norm: bool = False
