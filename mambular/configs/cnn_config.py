from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultCNNConfig:
    """
    Configuration class for the default CNN model with predefined hyperparameters.

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

    CNN Parameters
    --------------------
    input_channels : int, default=1
        Number of input channels (e.g., 1 for grayscale images).
    num_layers : int, default=4
        Number of convolutional layers.
    out_channels_list : list, default=(64, 64, 128, 128)
        List of output channels for each convolutional layer.
    kernel_size_list : list, default=(3, 3, 3, 3)
        List of kernel sizes for each convolutional layer.
    stride_list : list, default=(1, 1, 1, 1)
        List of stride values for each convolutional layer.
    padding_list : list, default=(1, 1, 1, 1)
        List of padding values for each convolutional layer.
    pooling_method : str, default="max"
        Pooling method ('max' or 'avg').
    pooling_kernel_size_list : list, default=(2, 2, 1, 1)
        List of kernel sizes for pooling layers for each convolutional layer.
    pooling_stride_list : list, default=(2, 2, 1, 1)
        List of stride values for pooling layers for each convolutional layer.

    Dropout Parameters
    -------------------
    dropout_rate : float, default=0.5
        Probability of dropping neurons during training.
    dropout_positions : list, default=None
        List of indices of layers after which dropout should be applied. If None, no dropout is applied.
    """

    # Optimizer parameters
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1

    # Embedding parameters
    layer_norm: bool = False
    layer_norm_eps: float = 1e-05
    use_embeddings: bool = False
    embedding_activation: callable = nn.Identity()
    embedding_type: str = "linear"
    embedding_bias: bool = False
    layer_norm_after_embedding: bool = False
    d_model: int = 32
    plr_lite: bool = False

    # CNN parameters
    input_channels: int = 1
    num_layers: int = 4
    out_channels_list: list = (64, 64, 128, 128)
    kernel_size_list: list = (3, 3, 3, 3)
    stride_list: list = (1, 1, 1, 1)
    padding_list: list = (1, 1, 1, 1)
    pooling_method: str = "max"
    pooling_kernel_size_list: list = (2, 2, 1, 1)
    pooling_stride_list: list = (2, 2, 1, 1)
    dropout_rate: float = 0.5  # Probability to drop neurons
    dropout_positions: list = None
