from dataclasses import dataclass
import torch.nn as nn
from typing import Literal


@dataclass
class DefaultTabMConfig:
    """
    Configuration class for the TabM model with batch ensembling and predefined hyperparameters.

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
    layer_sizes : list, default=(256, 128, 32)
        Sizes of the layers in the model.
    activation : callable, default=nn.SELU()
        Activation function for the model layers.
    skip_layers : bool, default=False
        Whether to skip layers in the model.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the model.
    skip_connections : bool, default=False
        Whether to use skip connections in the model.
    batch_norm : bool, default=False
        Whether to use batch normalization in the model layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the model layers.
    use_embeddings : bool, default=False
        Whether to use embedding layers for all features.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    d_model : int, default=32
        Dimensionality of the embeddings.
    ensemble_size : int, default=4
        Number of ensemble members for batch ensembling.
    ensemble_scaling_in : bool, default=True
        Whether to use input scaling for each ensemble member.
    ensemble_scaling_out : bool, default=True
        Whether to use output scaling for each ensemble member.
    ensemble_bias : bool, default=False
        Whether to use a unique bias term for each ensemble member.
    scaling_init : Literal['ones', 'random-signs'], default='ones'
        Initialization method for scaling weights.
    average_ensembles : bool, default=True
        Whether to average the outputs of the ensembles.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    layer_sizes: list = (256, 256, 32)
    activation: callable = nn.SELU()
    skip_layers: bool = False
    dropout: float = 0.5
    norm: str = None
    use_glu: bool = False
    skip_connections: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    layer_norm_eps: float = 1e-05

    # embedding params
    use_embeddings: bool = True
    embedding_type: float = "plr"
    plr_lite: bool = False
    average_embeddings: bool = True
    embedding_activation: callable = nn.Identity()
    layer_norm_after_embedding: bool = False
    d_model: int = 64

    # Batch ensembling specific configurations
    ensemble_size: int = 32
    ensemble_scaling_in: bool = True
    ensemble_scaling_out: bool = True
    ensemble_bias: bool = True
    scaling_init: Literal["ones", "random-signs", "normal"] = "normal"
    average_ensembles: bool = False
    model_type: Literal["mini", "full"] = "mini"
