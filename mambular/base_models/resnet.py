import torch
import torch.nn as nn
from typing import Any
from ..utils.configs import DefaultResNetConfig
from .basemodel import BaseModel
from ..utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)


class ResidualBlock(nn.Module):
    def __init__(
        self, input_dim, output_dim, activation, batch_norm, norm_layer, dropout
    ):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = batch_norm
        self.norm_layer = norm_layer
        self.activation = activation
        self.dropout = dropout

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        if norm_layer:
            layers.append(norm_layer(output_dim))
        layers.append(activation)
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.layers(out)
        out += identity
        return out


class ResNet(BaseModel):
    def __init__(
        self,
        cat_feature_info: Any,
        num_feature_info: Any,
        input_dim: int,
        num_classes: int = 1,
        config: DefaultResNetConfig = DefaultResNetConfig(),
        **kwargs,
    ):
        """
        Initializes the ResNet model with the given configuration.

        Parameters
        ----------
        cat_feature_info : Any
            Information about categorical features.
        num_feature_info : Any
            Information about numerical features.
        input_dim : int
            Number of input features.
        num_classes : int, optional
            Number of output classes, by default 1.
        config : DefaultResNetConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultResNetConfig().
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        self.activation = config.activation
        # Normalization layer
        norm_layer = self.hparams.get("norm", config.norm)
        if norm_layer == "RMSNorm":
            self.norm_f = RMSNorm
        elif norm_layer == "LayerNorm":
            self.norm_f = LayerNorm
        elif norm_layer == "BatchNorm":
            self.norm_f = BatchNorm
        elif norm_layer == "InstanceNorm":
            self.norm_f = InstanceNorm
        elif norm_layer == "GroupNorm":
            self.norm_f = GroupNorm
        elif norm_layer == "LearnableLayerScaling":
            self.norm_f = LearnableLayerScaling
        else:
            self.norm_f = None

        # Initial layer
        self.initial_layer = nn.Linear(input_dim, config.hidden_dims[0])

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(config.num_blocks):
            input_dim = (
                config.hidden_dims[i]
                if i < len(config.hidden_dims)
                else config.hidden_dims[-1]
            )
            output_dim = (
                config.hidden_dims[i + 1]
                if i + 1 < len(config.hidden_dims)
                else config.hidden_dims[-1]
            )
            block = ResidualBlock(
                input_dim,
                output_dim,
                self.activation,
                config.batch_norm,
                self.norm_f,
                config.dropout,
            )
            self.blocks.append(block)

        # Output layer
        self.output_layer = nn.Linear(config.hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x
