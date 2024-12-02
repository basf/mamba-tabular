from dataclasses import dataclass
import torch.nn as nn
from typing import Literal


@dataclass
class DefaultTREMConfig:
    """
    Configuration class for the Tabular Recurrent Ensemble Model (TREM)

    Attributes
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-05
        Weight decay (L2 regularization) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    d_model : int, default=128
        Dimensionality of the model.
    n_layers : int, default=4
        Number of RNN layers in the model.
    rnn_dropout : float, default=0.3
        Dropout rate for RNN layers.
    norm : str, default="RMSNorm"
        Type of normalization to be used ('RMSNorm', 'LayerNorm', etc.).
    activation : callable, default=nn.SELU()
        Activation function for the RNN model.
    embedding_activation : callable, default=nn.Identity()
        Activation function for numerical embeddings.
    embedding_dropout : float, optional
        Dropout rate applied to embeddings. If None, no dropout is applied.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after the embedding layer.
    pooling_method : str, default="avg"
        Pooling method to be used ('cls', 'avg', etc.).
    norm_first : bool, default=False
        Whether to apply normalization before other operations in each RNN block.
    bias : bool, default=True
        Whether to use bias in the linear layers.
    rnn_activation : callable, default=nn.ReLU()
        Activation function for RNN layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization to improve numerical stability.
    dim_feedforward : int, default=256
        Dimensionality of the feed-forward layers.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', 'plr'.).
    embedding_bias : bool, default=False
        Whether to have a bias in the embedding layer
    cat_encoding : str, default="int"
        Encoding method for categorical features ('int', 'one-hot', 'linear').
    d_conv : int, default=4
        Dimensionality of convolutional layers, if used.
    conv_bias : bool, default=True
        Whether to use bias in convolutional layers.
    residuals : bool, default=False
        Whether to include residual connections.

    Batch Ensembling Specific Attributes
    ------------------------------------
    ensemble_size : int, default=32
        Number of ensemble members in batch ensembling.
    ensemble_scaling_in : bool, default=True
        Whether to apply scaling to input features for each ensemble member.
    ensemble_scaling_out : bool, default=True
        Whether to apply scaling to outputs for each ensemble member.
    ensemble_bias : bool, default=True
        Whether to include bias for ensemble-specific scaling.
    scaling_init : {"ones", "random-signs", "normal"}, default="ones"
        Initialization method for ensemble scaling factors.
    average_ensembles : bool, default=False
        Whether to average predictions across ensemble members.
    model_type : {"mini", "full"}, default="mini"
        Model type to use ('mini' for reduced version, 'full' for complete model).
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-05
    lr_factor: float = 0.1
    d_model: int = 128
    n_layers: int = 4
    rnn_dropout: float = 0.3
    norm: str = "RMSNorm"
    activation: callable = nn.SELU()
    embedding_activation: callable = nn.Identity()
    embedding_dropout: float = None
    layer_norm_after_embedding: bool = False
    pooling_method: str = "avg"
    norm_first: bool = False
    bias: bool = True
    rnn_activation: callable = nn.ReLU()
    layer_norm_eps: float = 1e-05
    dim_feedforward: int = 256
    embedding_type: float = "linear"
    embedding_bias: bool = False
    cat_encoding: str = "int"
    d_conv: int = 4
    conv_bias: bool = True
    residuals: bool = False

    # Batch ensembling specific configurations
    ensemble_size: int = 32
    ensemble_scaling_in: bool = True
    ensemble_scaling_out: bool = True
    ensemble_bias: bool = True
    scaling_init: Literal["ones", "random-signs", "normal"] = "ones"
    average_ensembles: bool = False
    model_type: Literal["mini", "full"] = "mini"
