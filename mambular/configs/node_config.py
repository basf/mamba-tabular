from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultNODEConfig:
    """
    Configuration class for the default Neural Oblivious Decision Ensemble (NODE) model.

    This class provides default hyperparameters for training and configuring a NODE model.

    Attributes
    ----------
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-4.
    lr_patience : int, optional
        Number of epochs without improvement after which the learning rate will be reduced. Default is 10.
    weight_decay : float, optional
        Weight decay (L2 regularization penalty) applied by the optimizer. Default is 1e-6.
    lr_factor : float, optional
        Factor by which the learning rate is reduced when there is no improvement. Default is 0.1.
    norm : str, optional
        Type of normalization to use. Default is None.
    use_embeddings : bool, optional
        Whether to use embedding layers for categorical features. Default is False.
    embedding_activation : callable, optional
        Activation function to apply to embeddings. Default is `nn.Identity`.
    layer_norm_after_embedding : bool, optional
        Whether to apply layer normalization after embedding layers. Default is False.
    d_model : int, optional
        Dimensionality of the embedding space. Default is 32.
    num_layers : int, optional
        Number of dense layers in the model. Default is 4.
    layer_dim : int, optional
        Dimensionality of each dense layer. Default is 128.
    tree_dim : int, optional
        Dimensionality of the output from each tree leaf. Default is 1.
    depth : int, optional
        Depth of each decision tree in the ensemble. Default is 6.
    head_layer_sizes : list, default=(128, 64, 32)
        Sizes of the layers in the head of the model.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    norm: str = None
    use_embeddings: bool = False
    embedding_activation: callable = nn.Identity()
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
