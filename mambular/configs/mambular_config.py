from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultMambularConfig:
    """
    Configuration class for the Default Mambular model with predefined hyperparameters.

    Optimizer Parameters
    --------------------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.

    Mamba Model Parameters
    -----------------------
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=4
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    dropout : float, default=0.0
        Dropout rate for regularization.
    dt_rank : str, default="auto"
        Rank of the decision tree used in the model.
    d_state : int, default=128
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree parameters.
    dt_init : str, default="random"
        Initialization method for decision tree parameters.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="LayerNorm"
        Type of normalization used ('LayerNorm', 'RMSNorm', etc.).
    activation : callable, default=nn.SiLU()
        Activation function for the model.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    AD_weight_decay : bool, default=True
        Whether weight decay is applied to A-D matrices.
    BC_layer_norm : bool, default=False
        Whether to apply layer normalization to B-C matrices.

    Embedding Parameters
    ---------------------
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    shuffle_embeddings : bool, default=False
        Whether to shuffle embeddings before being passed to Mamba layers.
    cat_encoding : str, default="int"
        Encoding method for categorical features ('int', 'one-hot', etc.).

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

    Additional Features
    --------------------
    pooling_method : str, default="avg"
        Pooling method to use ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to process data bidirectionally.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through Mamba blocks.
    use_cls : bool, default=False
        Whether to append a CLS token to the input sequences.
    use_pscan : bool, default=False
        Whether to use PSCAN for the state-space model.

    Mamba Version
    -------------
    mamba_version : str, default="mamba-torch"
        Version of the Mamba model to use ('mamba-torch', 'mamba1', 'mamba2').
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    d_model: int = 64
    n_layers: int = 4
    expand_factor: int = 2
    bias: bool = False
    d_conv: int = 4
    conv_bias: bool = True
    dropout: float = 0.0
    dt_rank: str = "auto"
    d_state: int = 128
    dt_scale: float = 1.0
    dt_init: str = "random"
    dt_max: float = 0.1
    dt_min: float = 1e-04
    dt_init_floor: float = 1e-04
    norm: str = "RMSNorm"
    activation: callable = nn.SiLU()
    embedding_activation: callable = nn.Identity()
    embedding_type: str = "linear"
    embedding_bias: bool = False
    head_layer_sizes: list = ()
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: callable = nn.SELU()
    head_use_batch_norm: bool = False
    layer_norm_after_embedding: bool = False
    pooling_method: str = "avg"
    bidirectional: bool = False
    use_learnable_interaction: bool = False
    use_cls: bool = False
    shuffle_embeddings: bool = False
    layer_norm_eps: float = 1e-05
    AD_weight_decay: bool = True
    BC_layer_norm: bool = False
    cat_encoding: str = "int"
    use_pscan: bool = False
    mamba_version: str = "mamba-torch"
