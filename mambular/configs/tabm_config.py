from dataclasses import dataclass
import torch.nn as nn
from typing import Literal


@dataclass
class DefaultTabMConfig:
    """
    Configuration class for the TabM model with batch ensembling and predefined hyperparameters.

    Optimizer Parameters
    --------------------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate is reduced when there is no improvement.

    Architecture Parameters
    ------------------------
    layer_sizes : list, default=(512, 512, 128)
        Sizes of the layers in the model.
    activation : callable, default=nn.ReLU()
        Activation function for the model layers.
    dropout : float, default=0.3
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the model.
    batch_norm : bool, default=False
        Whether to use batch normalization in the model layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the model layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.

    Embedding Parameters
    ---------------------
    use_embeddings : bool, default=True
        Whether to use embedding layers for all features.
    embedding_type : str, default="plr"
        Type of embedding to use ('plr', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    plr_lite : bool, default=False
        Whether to use a lightweight version of Piecewise Linear Regression (PLR).
    average_embeddings : bool, default=False
        Whether to average embeddings during the forward pass.
    embedding_activation : callable, default=nn.ReLU()
        Activation function for embeddings.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding layers.
    d_model : int, default=64
        Dimensionality of the embeddings.

    Batch Ensembling Parameters
    ----------------------------
    ensemble_size : int, default=32
        Number of ensemble members for batch ensembling.
    ensemble_scaling_in : bool, default=True
        Whether to use input scaling for each ensemble member.
    ensemble_scaling_out : bool, default=True
        Whether to use output scaling for each ensemble member.
    ensemble_bias : bool, default=True
        Whether to use a unique bias term for each ensemble member.
    scaling_init : {"ones", "random-signs", "normal"}, default="normal"
        Initialization method for scaling weights.
    average_ensembles : bool, default=False
        Whether to average the outputs of the ensembles.
    model_type : {"mini", "full"}, default="mini"
        Model type to use ('mini' for reduced version, 'full' for complete model).
    """

    # lr params
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-05
    lr_factor: float = 0.1

    # arch params
    layer_sizes: list = (256, 256, 128)
    activation: callable = nn.ReLU()
    dropout: float = 0.5
    norm: str = None
    use_glu: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    layer_norm_eps: float = 1e-05

    # embedding params
    use_embeddings: bool = True
    embedding_type: float = "plr"
    embedding_bias = False
    plr_lite: bool = False
    average_embeddings: bool = False
    embedding_activation: callable = nn.Identity()
    layer_norm_after_embedding: bool = False
    d_model: int = 32

    # Batch ensembling specific configurations
    ensemble_size: int = 32
    ensemble_scaling_in: bool = True
    ensemble_scaling_out: bool = True
    ensemble_bias: bool = True
    scaling_init: Literal["ones", "random-signs", "normal"] = "ones"
    average_ensembles: bool = False
    model_type: Literal["mini", "full"] = "mini"
