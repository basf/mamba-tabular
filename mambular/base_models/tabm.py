import torch
import torch.nn as nn
from ..configs.tabm_config import DefaultTabMConfig
from .basemodel import BaseModel
from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.layer_utils.batch_ensemble_layer import LinearBatchEnsembleLayer


class TabM(BaseModel):
    """
    A TabM model for tabular data, integrating feature embeddings, batch ensemble layers, and configurable
    architecture for processing categorical and numerical features with options for skip connections, GLU activation,
    and dropout.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultTabMConfig, optional
        Configuration object containing model hyperparameters such as layer sizes, dropout rates, batch ensemble
        settings, activation functions, and normalization settings, by default DefaultTabMConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    layer_sizes : list of int
        List specifying the number of units in each layer of the TabM model.
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    config : DefaultTabMConfig
        Stores the configuration for the TabM model.
    layers : nn.ModuleList
        List containing the layers of the TabM model, including LinearBatchEnsembleLayer, normalization, activation,
        and dropout layers.
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
        Normalization layer applied in each batch ensemble layer, if specified in the configuration.
    final_layer : nn.Linear, optional
        Final linear layer applied when ensemble outputs are not averaged.

    Methods
    -------
    forward(num_features, cat_features) -> torch.Tensor
        Perform a forward pass through the model, including embedding (if enabled), batch ensemble layers,
        optional skip connections, and prediction steps.

    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultTabMConfig = DefaultTabMConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.layer_sizes = self.hparams.get("layer_sizes", config.layer_sizes)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.config = config

        # Initialize layers
        self.layers = nn.ModuleList()
        self.skip_connections = self.hparams.get(
            "skip_connections", config.skip_connections
        )
        self.use_glu = self.hparams.get("use_glu", config.use_glu)
        self.activation = self.hparams.get("activation", config.activation)
        self.use_embeddings = self.hparams.get("use_embeddings", config.use_embeddings)

        # Embedding layer
        if self.use_embeddings:
            self.embedding_layer = EmbeddingLayer(config)
            if self.hparams.get("average_embeddings", config.average_embeddings):
                input_dim = self.hparams.get("d_model", config.d_model)
            else:
                input_dim = (
                    len(num_feature_info) + len(cat_feature_info)
                ) * config.d_model
                print(input_dim)

        else:
            # Calculate input dimension
            input_dim = sum(input_shape for input_shape in num_feature_info.values())
            input_dim += len(cat_feature_info)

        # Input layer with batch ensembling
        self.layers.append(
            LinearBatchEnsembleLayer(
                in_features=input_dim,
                out_features=self.layer_sizes[0],
                ensemble_size=config.ensemble_size,
                ensemble_scaling_in=config.ensemble_scaling_in,
                ensemble_scaling_out=config.ensemble_scaling_out,
                ensemble_bias=config.ensemble_bias,
                scaling_init=config.scaling_init,
            )
        )
        if config.batch_norm:
            self.layers.append(nn.BatchNorm1d(self.layer_sizes[0]))

        self.norm_f = get_normalization_layer(config)
        if self.norm_f is not None:
            self.layers.append(self.norm_f(self.layer_sizes[0]))

        # Optional activation and dropout
        if config.use_glu:
            self.layers.append(nn.GLU())
        else:
            self.layers.append(self.activation)
        if config.dropout > 0.0:
            self.layers.append(nn.Dropout(config.dropout))

        # Hidden layers with batch ensembling
        for i in range(1, len(self.layer_sizes)):
            self.layers.append(
                LinearBatchEnsembleLayer(
                    in_features=self.layer_sizes[i - 1],
                    out_features=self.layer_sizes[i],
                    ensemble_size=config.ensemble_size,
                    ensemble_scaling_in=config.ensemble_scaling_in,
                    ensemble_scaling_out=config.ensemble_scaling_out,
                    ensemble_bias=config.ensemble_bias,
                    scaling_init=config.scaling_init,
                )
            )
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
        self.layers.append(
            LinearBatchEnsembleLayer(
                in_features=self.layer_sizes[-1],
                out_features=num_classes,
                ensemble_size=config.ensemble_size,
                ensemble_scaling_in=config.ensemble_scaling_in,
                ensemble_scaling_out=config.ensemble_scaling_out,
                ensemble_bias=config.ensemble_bias,
                scaling_init=config.scaling_init,
            )
        )

        if not self.hparams.get("average_ensembles", True):
            self.final_layer = nn.Linear(
                self.layer_sizes[-1] * config.ensemble_size, num_classes
            )

    def forward(self, num_features, cat_features) -> torch.Tensor:
        """
        Forward pass of the TabM model with batch ensembling.

        Parameters
        ----------
        num_features : torch.Tensor
            Numerical features tensor.
        cat_features : torch.Tensor
            Categorical features tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # Handle embeddings if used
        if self.use_embeddings:
            x = self.embedding_layer(num_features, cat_features)
            # Option 1: Average over feature dimension (N)
            if self.hparams.get("average_embeddings", self.config.average_embeddings):
                x = x.mean(dim=1)  # Shape: (B, D)
            # Option 2: Flatten feature and embedding dimensions
            else:
                B, N, D = x.shape
                x = x.reshape(B, N * D)  # Shape: (B, N * D)

        else:
            x = num_features + cat_features
            x = torch.cat(x, dim=1)

        # Process through layers with optional skip connections
        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], LinearBatchEnsembleLayer):
                out = self.layers[i](x)
                # `out` shape is expected to be (batch_size, ensemble_size, out_features)
                if self.skip_connections and x.shape == out.shape:
                    x = x + out
                else:
                    x = out
            else:
                x = self.layers[i](x)

        # Final ensemble output from the last ConfigurableBatchEnsembleLayer
        x = self.layers[-1](x)  # Shape (batch_size, ensemble_size, num_classes)

        # Option 1: Averaging across ensemble outputs
        if self.hparams.get("average_ensembles", True):
            x = x.mean(dim=1)  # Shape (batch_size, num_classes)

        # Option 2: Adding a final layer to map to `num_classes`
        else:
            x = x.view(x.size(0), -1)  # Flatten ensemble dimension if not averaging
            x = self.final_layer(x)  # Shape (batch_size, num_classes)

        return x
