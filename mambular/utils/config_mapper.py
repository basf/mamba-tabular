from skopt.space import Real, Integer, Categorical
import torch.nn as nn
from ..arch_utils.transformer_utils import ReGLU


def round_to_nearest_16(x):
    """Rounds the value to the nearest multiple of 16."""
    return int(round(x / 16) * 16)


def get_search_space(config):
    """
    Given a model configuration, return the hyperparameter search space
    based on the config attributes.

    Parameters
    ----------
    config : dataclass
        The configuration object for the model.

    Returns
    -------
    param_names : list
        A list of parameter names to be optimized.
    param_space : list
        A list of hyperparameter ranges for Bayesian optimization.
    """

    search_space_mapping = {
        # Learning rate-related parameters
        "lr": Real(1e-6, 1e-2, prior="log-uniform"),
        "lr_patience": Integer(5, 20),
        "lr_factor": Real(0.1, 0.5),
        # Model architecture parameters
        "n_layers": Integer(1, 8),
        "d_model": Integer(16, 512),  # Dimension of the model
        "dropout": Real(0.0, 0.5),
        "expand_factor": Integer(1, 4),
        "d_state": Integer(16, 512),
        "ff_dropout": Real(0.0, 0.5),
        "rnn_dropout": Real(0.0, 0.5),
        "attn_dropout": Real(0.0, 0.5),
        "n_heads": Integer(1, 8),
        "transformer_dim_feedforward": Integer(16, 512),
        # Convolution-related parameters
        "d_conv": Integer(4, 128),  # Dimension of convolution layers
        "conv_bias": Categorical([True, False]),
        # Normalization and regularization
        "norm": Categorical(["LayerNorm", "BatchNorm", "RMSNorm"]),
        "weight_decay": Real(1e-8, 1e-2, prior="log-uniform"),
        "layer_norm_eps": Real(1e-7, 1e-4),
        "head_dropout": Real(0.0, 0.5),
        "bias": Categorical([True, False]),
        "norm_first": Categorical([True, False]),
        # Pooling, activation, and head layer settings
        "pooling_method": Categorical(["avg", "max", "cls", "sum"]),
        "activation": Categorical(
            ["ReLU", "SELU", "Identity", "Tanh", "LeakyReLU", "SiLU"]
        ),
        "embedding_activation": Categorical(
            ["ReLU", "SELU", "Identity", "Tanh", "LeakyReLU"]
        ),
        "rnn_activation": Categorical(["relu", "tanh"]),
        "transformer_activation": Categorical(
            ["ReLU", "SELU", "Identity", "Tanh", "LeakyReLU", "ReGLU"]
        ),
        "head_skip_layers": Categorical([True, False]),
        "head_use_batch_norm": Categorical([True, False]),
        # Sequence-related settings
        "bidirectional": Categorical([True, False]),
        "use_learnable_interaction": Categorical([True, False]),
        "use_cls": Categorical([True, False]),
        # Feature encoding
        "cat_encoding": Categorical(["int", "one-hot"]),
    }

    layer_size_min, layer_size_max = 16, 512  # Dynamic layer sizes
    max_head_layers = 5  # Set a maximum number of layers for optimization

    param_names = []
    param_space = []

    for field in config.__dataclass_fields__:
        if field in search_space_mapping:
            param_names.append(field)
            param_space.append(search_space_mapping[field])

    # Handle head_layer_sizes dynamically by setting the length and individual sizes
    if "head_layer_sizes" in config.__dataclass_fields__:
        param_names.append("head_layer_size_length")
        param_space.append(
            Integer(1, max_head_layers)
        )  # Optimize the length of the list

        # Optimize individual layer sizes based on max_head_layers
        for i in range(max_head_layers):
            # Optimize over integers and multiply by 16 to ensure divisibility by 16
            param_names.append(f"head_layer_size_{i+1}")
            param_space.append(Integer(layer_size_min, layer_size_max))

    return param_names, param_space


activation_mapper = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "SiLU": nn.SiLU(),
    "LeakyReLU": nn.LeakyReLU(),
    "Identity": nn.Identity(),
    "Linear": nn.Identity(),
    "SELU": nn.SELU(),
    "ReGLU": ReGLU(),
}
