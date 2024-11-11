import torch
import torch.nn as nn
from ..configs.mlp_config import DefaultMLPConfig
from .basemodel import BaseModel
from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer


class MLP(BaseModel):
    """
    A multi-layer perceptron (MLP) model for tabular data processing, with options for embedding, normalization,
    skip connections, and customizable activation functions.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultMLPConfig, optional
        Configuration object with model hyperparameters such as layer sizes, dropout rates, activation functions,
        embedding settings, and normalization options, by default DefaultMLPConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    layer_sizes : list of int
        List specifying the number of units in each layer of the MLP.
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    layers : nn.ModuleList
        List containing the layers of the MLP, including linear layers, normalization layers, and activations.
    skip_connections : bool
        Flag indicating whether skip connections are enabled between layers.
    use_glu : bool
        Flag indicating if gated linear units (GLU) should be used as the activation function.
    activation : callable
        Activation function applied between layers.
    use_embeddings : bool
        Flag indicating if embeddings should be used for categorical and numerical features.
    embedding_layer : EmbeddingLayer, optional
        Embedding layer for features, used if `use_embeddings` is enabled.
    norm_f : nn.Module, optional
        Normalization layer applied to the output of the first layer, if specified in the configuration.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including embedding (if enabled), linear transformations,
        activation, normalization, and prediction steps.

    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultMLPConfig = DefaultMLPConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.layer_sizes = self.hparams.get("layer_sizes", config.layer_sizes)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        # Initialize layers
        self.layers = nn.ModuleList()
        self.skip_connections = self.hparams.get(
            "skip_connections", config.skip_connections
        )
        self.use_glu = self.hparams.get("use_glu", config.use_glu)
        self.activation = self.hparams.get("activation", config.activation)
        self.use_embeddings = self.hparams.get("use_embeddings", config.use_embeddings)

        input_dim = 0
        for feature_name, input_shape in num_feature_info.items():
            input_dim += input_shape
        for feature_name, input_shape in cat_feature_info.items():
            input_dim += 1

        if self.use_embeddings:
            self.embedding_layer = EmbeddingLayer(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
                config=config,
            )
            input_dim = (
                len(num_feature_info) * config.d_model
                + len(cat_feature_info) * config.d_model
            )

        # Input layer
        self.layers.append(nn.Linear(input_dim, self.layer_sizes[0]))
        if config.batch_norm:
            self.layers.append(nn.BatchNorm1d(self.layer_sizes[0]))

        self.norm_f = get_normalization_layer(config)

        if self.norm_f is not None:
            self.layers.append(self.norm_f(self.layer_sizes[0]))

        if config.use_glu:
            self.layers.append(nn.GLU())
        else:
            self.layers.append(self.activation)
        if config.dropout > 0.0:
            self.layers.append(nn.Dropout(config.dropout))

        # Hidden layers
        for i in range(1, len(self.layer_sizes)):
            self.layers.append(nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i]))
            if config.batch_norm:
                self.layers.append(nn.BatchNorm1d(self.layer_sizes[i]))
            if config.layer_norm:
                self.layers.append(nn.LayerNorm(self.layer_sizes[i]))
            if config.use_glu:
                self.layers.append(nn.GLU())
            else:
                self.layers.append(self.activation)
            if config.dropout > 0.0:
                self.layers.append(nn.Dropout(config.dropout))

        # Output layer
        self.layers.append(nn.Linear(self.layer_sizes[-1], num_classes))

    def forward(self, num_features, cat_features) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.use_embeddings:
            x = self.embedding_layer(num_features, cat_features)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
        else:
            x = num_features + cat_features
            x = torch.cat(x, dim=1)

        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], nn.Linear):
                out = self.layers[i](x)
                if self.skip_connections and x.shape == out.shape:
                    x = x + out
                else:
                    x = out
            else:
                x = self.layers[i](x)

        x = self.layers[-1](x)
        return x
