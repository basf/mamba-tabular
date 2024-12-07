from dataclasses import dataclass
import torch.nn as nn
from typing import Literal


@dataclass
class DefaultTabulaRNNConfig:
    """
    Configuration class for the TabulaRNN model with predefined hyperparameters.

    Optimizer Parameters
    --------------------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.

    Architecture Parameters
    ------------------------
    model_type : str, default="RNN"
        Type of model, one of "RNN", "LSTM", "GRU", "mLSTM", "sLSTM".
    d_model : int, default=128
        Dimensionality of the model.
    n_layers : int, default=4
        Number of layers in the RNN.
    rnn_dropout : float, default=0.2
        Dropout rate for the RNN layers.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the RNN layers.
    residuals : bool, default=False
        Whether to include residual connections in the RNN.

    Embedding Parameters
    ---------------------
    embedding_type : str, default="linear"
        Type of embedding for features ('linear', 'plr', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    plr_lite : bool, default=False
        Whether to use a lightweight version of Piecewise Linear Regression (PLR).
    n_frequencies : int, default=48
        Number of frequencies for PLR embeddings.
    frequencies_init_scale : float, default=0.01
        Initial scale for frequency parameters in embeddings.
    embedding_activation : callable, default=nn.ReLU()
        Activation function for embeddings.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding layers.

    Head Parameters
    ----------------
    head_layer_sizes : list, default=()
        Sizes of the layers in the head of the model.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.

    Pooling and Normalization
    --------------------------
    pooling_method : str, default="avg"
        Pooling method to be used ('avg', 'cls', etc.).
    norm_first : bool, default=False
        Whether to apply normalization before other operations in each block.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.

    Additional Parameters
    ----------------------
    bias : bool, default=True
        Whether to use bias in the linear layers.
    rnn_activation : str, default="relu"
        Activation function for the RNN layers.
    dim_feedforward : int, default=256
        Size of the feedforward network.
    d_conv : int, default=4
        Size of the convolutional layer for embedding features.
    conv_bias : bool, default=True
        Whether to use bias in the convolutional layers.
    """

    # Optimizer params
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1

    # Architecture params
    model_type: str = "RNN"
    d_model: int = 128
    n_layers: int = 4
    rnn_dropout: float = 0.2
    norm: str = "RMSNorm"
    activation: callable = nn.SELU()
    residuals: bool = False

    # Embedding params
    embedding_type: str = "linear"
    embedding_bias: bool = False
    plr_lite: bool = False
    n_frequencies: int = 48
    frequencies_init_scale: float = 0.01
    embedding_activation: callable = nn.ReLU()
    layer_norm_after_embedding: bool = False

    # Head params
    head_layer_sizes: list = ()
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: callable = nn.SELU()
    head_use_batch_norm: bool = False

    # Pooling and normalization
    pooling_method: str = "avg"
    norm_first: bool = False
    layer_norm_eps: float = 1e-05

    # Additional params
    bias: bool = True
    rnn_activation: str = "relu"
    dim_feedforward: int = 256
    d_conv: int = 4
    conv_bias: bool = True
