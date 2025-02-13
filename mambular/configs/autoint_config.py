from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Optional
import torch.nn as nn
from .base_config import BaseConfig


@dataclass
class DefaultAutoIntConfig(BaseConfig):
    """Configuration class for the AutoInt model with predefined hyperparameters.

    Parameters
    ----------
    n_layers : int, default=6
        Number of transformer layers in the model.
    n_heads : int, default=8
        Number of attention heads used in the multi-head self-attention mechanism.
    attention_dropout : float, default=0.1
        Dropout rate applied within the attention mechanism.
    residual_dropout : float, default=0.1
        Dropout rate applied to residual connections after attention layers.
    activation : str, default="relu"
        Activation function used in the transformer layers.
    prenormalization : bool, default=False
        Whether to apply normalization before attention layers instead of after.
    norm_first : bool, default=True
        Whether to apply normalization before each sub-layer (pre-norm).
    norm : str, default="LayerNorm"
        Type of normalization to be used ('LayerNorm', 'BatchNorm', etc.).
    kv_compression : Optional[float], default=None
        Compression ratio for key-value vectors. If None, no compression is applied.
    kv_compression_sharing : Optional[str], default=None
        How key-value compression is shared across layers. Options:
        - "layerwise": Shared across all layers.
        - "headwise": Different compression per head.
        - "key-value": Separate compression for keys and values.
    pooling_method : str, default="avg"
        Pooling strategy applied after transformer layers. Options:
        - "avg": Average pooling over token embeddings.
        - "cls": Use a [CLS] token representation.
    use_cls : bool, default=False
        Whether to append a [CLS] token and use it for final classification.
    cat_encoding : str, default="int"
        Encoding method for categorical features. Options:
        - "int": Integer encoding.
        - "one-hot": One-hot encoding.
        - "embedding": Trainable embeddings.
    head_layer_sizes : list, default=()
        Sizes of the fully connected layers in the model's head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to use skip connections in the head layers.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    """

    # Transformer Parameters
    n_layers: int = 6  # Number of transformer layers
    n_heads: int = 8  # Number of attention heads
    attention_dropout: float = 0.1  # Dropout in attention mechanism
    residual_dropout: float = 0.1  # Dropout in residual connections

    # Activation & Normalization
    activation: Callable = nn.ReLU()  # Activation function
    prenormalization: bool = False  # Whether to apply prenormalization
    norm_first: bool = True
    norm: str = "LayerNorm"

    # Key-Value Compression
    kv_compression: Optional[float] = None  # Compression ratio for key-value vectors
    kv_compression_sharing: Optional[str] = (
        None  # Compression sharing mode ('layerwise', 'headwise', 'key-value')
    )

    # Pooling and Categorical Encoding
    pooling_method: str = "avg"
    use_cls: bool = False
    cat_encoding: str = "int"

    # Head Parameters
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False
