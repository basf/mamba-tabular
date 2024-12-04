import torch
import torch.nn as nn
from typing import Any
from ..configs.resnet_config import DefaultResNetConfig
from .basemodel import BaseModel
from ..arch_utils.resnet_utils import ResidualBlock
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..utils.get_feature_dimensions import get_feature_dimensions


class ResNet(BaseModel):
    """
    A ResNet model for tabular data, combining feature embeddings, residual blocks, and customizable architecture
    for processing categorical and numerical features.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultResNetConfig, optional
        Configuration object containing model hyperparameters such as layer sizes, number of residual blocks,
        dropout rates, activation functions, and normalization settings, by default DefaultResNetConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    layer_sizes : list of int
        List specifying the number of units in each layer of the ResNet.
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    activation : callable
        Activation function used in the residual blocks.
    use_embeddings : bool
        Flag indicating if embeddings should be used for categorical and numerical features.
    embedding_layer : EmbeddingLayer, optional
        Embedding layer for features, used if `use_embeddings` is enabled.
    initial_layer : nn.Linear
        Initial linear layer to project input features into the model's hidden dimension.
    blocks : nn.ModuleList
        List of residual blocks to process the hidden representations.
    output_layer : nn.Linear
        Output layer that produces the final prediction.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including embedding (if enabled), residual blocks,
        and prediction steps.

    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultResNetConfig = DefaultResNetConfig(),
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.returns_ensemble = False
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        if self.hparams.use_embeddings:
            input_dim = (
                len(num_feature_info) * self.hparams.d_model
                + len(cat_feature_info) * self.hparams.d_model
            )
            # embedding layer
            self.embedding_layer = EmbeddingLayer(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
                config=config,
            )

        else:
            input_dim = get_feature_dimensions(num_feature_info, cat_feature_info)

        self.initial_layer = nn.Linear(input_dim, self.hparams.layer_sizes[0])

        self.blocks = nn.ModuleList()
        for i in range(self.hparams.num_blocks):
            input_dim = self.hparams.layer_sizes[i]
            output_dim = (
                self.hparams.layer_sizes[i + 1]
                if i + 1 < len(self.hparams.layer_sizes)
                else self.hparams.layer_sizes[-1]
            )
            block = ResidualBlock(
                input_dim,
                output_dim,
                self.hparams.activation,
                self.hparams.norm,
                self.hparams.dropout,
            )
            self.blocks.append(block)

        self.output_layer = nn.Linear(self.hparams.layer_sizes[-1], num_classes)

    def forward(self, num_features, cat_features):
        if self.hparams.use_embeddings:
            x = self.embedding_layer(num_features, cat_features)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
        else:
            x = num_features + cat_features
            x = torch.cat(x, dim=1)

        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x
