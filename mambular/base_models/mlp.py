import torch
import torch.nn as nn
from ..utils.mlp_utils import Linear_block, Linear_skip_block
from ..utils.configs import DefaultMLPConfig
from .basemodel import BaseModel
import numpy as np
from ..utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)


class MLP(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        input_dim: int,
        num_classes: int = 1,
        config: DefaultMLPConfig = DefaultMLPConfig(),
        **kwargs,
    ):
        """
        Initializes the MLP model with the given configuration.

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
        config : DefaultMLPConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultMLPConfig().
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        # Initialize layers
        self.layers = nn.ModuleList()
        self.skip_connections = config.skip_connections
        self.use_attention = config.use_attention
        self.use_glu = config.use_glu
        self.activation = config.activation

        # Input layer
        self.layers.append(nn.Linear(input_dim, config.hidden_dims[0]))
        if config.batch_norm:
            self.layers.append(nn.BatchNorm1d(config.hidden_dims[0]))

        norm_layer = self.hparams.get("norm", config.norm)
        if norm_layer == "RMSNorm":
            self.norm_f = RMSNorm(config.hidden_dims[0])
        elif norm_layer == "LayerNorm":
            self.norm_f = LayerNorm(config.hidden_dims[0])
        elif norm_layer == "BatchNorm":
            self.norm_f = BatchNorm(config.hidden_dims[0])
        elif norm_layer == "InstanceNorm":
            self.norm_f = InstanceNorm(config.hidden_dims[0])
        elif norm_layer == "GroupNorm":
            self.norm_f = GroupNorm(1, config.hidden_dims[0])
        elif norm_layer == "LearnableLayerScaling":
            self.norm_f = LearnableLayerScaling(config.hidden_dims[0])
        else:
            self.norm_f = None

        if self.norm_f is not None:
            self.layers.append(self.norm_f(config.hidden_dims[0]))

        if config.use_glu:
            self.layers.append(nn.GLU())
        else:
            self.layers.append(self.activation)
        if config.dropout > 0.0:
            self.layers.append(nn.Dropout(config.dropout))

        # Hidden layers
        for i in range(1, len(config.hidden_dims)):
            self.layers.append(
                nn.Linear(config.hidden_dims[i - 1], config.hidden_dims[i])
            )
            if config.batch_norm:
                self.layers.append(nn.BatchNorm1d(config.hidden_dims[i]))
            if config.layer_norm:
                self.layers.append(nn.LayerNorm(config.hidden_dims[i]))
            if config.use_glu:
                self.layers.append(nn.GLU())
            else:
                self.layers.append(self.activation)
            if config.dropout > 0.0:
                self.layers.append(nn.Dropout(config.dropout))

        # Output layer
        self.layers.append(nn.Linear(config.hidden_dims[-1], num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
