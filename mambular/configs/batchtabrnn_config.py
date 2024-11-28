from dataclasses import dataclass
import torch.nn as nn
from typing import Literal


@dataclass
class DefaultBatchTabRNNConfig:
    """
    Configuration class for the default TabulaRNN model with predefined hyperparameters.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    model_type : str, default="RNN"
        type of model, one of "RNN", "LSTM", "GRU", "mLSTM", "sLSTM"
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=8
        Number of layers in the transformer.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the transformer.
    embedding_activation : callable, default=nn.Identity()
        Activation function for numerical embeddings.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    pooling_method : str, default="cls"
        Pooling method to be used ('cls', 'avg', etc.).
    norm_first : bool, default=False
        Whether to apply normalization before other operations in each transformer block.
    bias : bool, default=True
        Whether to use bias in the linear layers.
    rnn_activation : callable, default=nn.SELU()
        Activation function for the transformer layers.
    bidirectional : bool, default=False.
        Whether to process data bidirectionally
    cat_encoding : str, default="int"
        Encoding method for categorical features.
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
    embedding_type: float = "standard"
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
