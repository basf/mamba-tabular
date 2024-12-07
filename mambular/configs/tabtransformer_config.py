from dataclasses import dataclass
import torch.nn as nn
from ..arch_utils.transformer_utils import ReGLU


@dataclass
class DefaultTabTransformerConfig:
    """
    Configuration class for the default Tab Transformer model with predefined hyperparameters.

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
    d_model : int, default=128
        Dimensionality of the model.
    n_layers : int, default=4
        Number of layers in the transformer.
    n_heads : int, default=8
        Number of attention heads in the transformer.
    attn_dropout : float, default=0.2
        Dropout rate for the attention mechanism.
    ff_dropout : float, default=0.1
        Dropout rate for the feed-forward layers.
    norm : str, default="LayerNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the transformer layers.
    transformer_activation : callable, default=ReGLU()
        Activation function for the transformer layers.
    transformer_dim_feedforward : int, default=512
        Dimensionality of the feed-forward layers in the transformer.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    norm_first : bool, default=True
        Whether to apply normalization before other operations in each transformer block.
    bias : bool, default=True
        Whether to use bias in the linear layers.

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

    Pooling and Categorical Encoding
    ---------------------------------
    pooling_method : str, default="avg"
        Pooling method to be used ('cls', 'avg', etc.).
    cat_encoding : str, default="int"
        Encoding method for categorical features ('int', 'one-hot', etc.).
    """

    # Optimizer Parameters
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1

    # Architecture Parameters
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    attn_dropout: float = 0.2
    ff_dropout: float = 0.1
    norm: str = "LayerNorm"
    activation: callable = nn.SELU()
    transformer_activation: callable = ReGLU()
    transformer_dim_feedforward: int = 512
    layer_norm_eps: float = 1e-05
    norm_first: bool = True
    bias: bool = True

    # Embedding Parameters
    embedding_activation: callable = nn.Identity()
    embedding_type: str = "linear"
    embedding_bias: bool = False
    layer_norm_after_embedding: bool = False

    # Head Parameters
    head_layer_sizes: list = ()
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: callable = nn.SELU()
    head_use_batch_norm: bool = False

    # Pooling and Categorical Encoding
    pooling_method: str = "avg"
    cat_encoding: str = "int"
