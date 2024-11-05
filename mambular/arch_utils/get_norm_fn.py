from .layer_utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)


def get_normalization_layer(config):
    """
    Function to return the appropriate normalization layer based on the configuration.

    Parameters:
    -----------
    config : DefaultMambularConfig
        Configuration object containing the parameters for the model including normalization.

    Returns:
    --------
    nn.Module:
        The normalization layer as per the config.

    Raises:
    -------
    ValueError:
        If an unsupported normalization layer is specified in the config.
    """

    norm_layer = config.norm

    d_model = config.d_model
    layer_norm_eps = config.layer_norm_eps

    if norm_layer == "RMSNorm":
        return RMSNorm(d_model, eps=layer_norm_eps)
    elif norm_layer == "LayerNorm":
        return LayerNorm(d_model, eps=layer_norm_eps)
    elif norm_layer == "BatchNorm":
        return BatchNorm(d_model, eps=layer_norm_eps)
    elif norm_layer == "InstanceNorm":
        return InstanceNorm(d_model, eps=layer_norm_eps)
    elif norm_layer == "GroupNorm":
        return GroupNorm(1, d_model, eps=layer_norm_eps)
    elif norm_layer == "LearnableLayerScaling":
        return LearnableLayerScaling(d_model)
    else:
        raise ValueError(f"Unsupported normalization layer: {norm_layer}")
