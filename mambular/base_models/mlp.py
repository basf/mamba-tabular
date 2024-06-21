import torch
import torch.nn as nn
from ..configs.mlp_config import DefaultMLPConfig
from .basemodel import BaseModel
from ..arch_utils.normalization_layers import (
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

        num_classes : int, optional
            Number of output classes, by default 1.
        config : DefaultMLPConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultMLPConfig().
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        input_dim = 0
        for feature_name, input_shape in num_feature_info.items():
            input_dim += input_shape
        for feature_name, input_shape in cat_feature_info.items():
            input_dim += 1

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        # Initialize layers
        self.layers = nn.ModuleList()
        self.skip_connections = self.hparams.get(
            "skip_connections", config.skip_connections
        )
        self.use_glu = self.hparams.get("use_glu", config.use_glu)
        self.activation = self.hparams.get("activation", config.activation)

        # Input layer
        self.layers.append(nn.Linear(input_dim, config.layer_sizes[0]))
        if config.batch_norm:
            self.layers.append(nn.BatchNorm1d(config.layer_sizes[0]))

        norm_layer = self.hparams.get("norm", config.norm)
        if norm_layer == "RMSNorm":
            self.norm_f = RMSNorm(config.layer_sizes[0])
        elif norm_layer == "LayerNorm":
            self.norm_f = LayerNorm(config.layer_sizes[0])
        elif norm_layer == "BatchNorm":
            self.norm_f = BatchNorm(config.layer_sizes[0])
        elif norm_layer == "InstanceNorm":
            self.norm_f = InstanceNorm(config.layer_sizes[0])
        elif norm_layer == "GroupNorm":
            self.norm_f = GroupNorm(1, config.layer_sizes[0])
        elif norm_layer == "LearnableLayerScaling":
            self.norm_f = LearnableLayerScaling(config.layer_sizes[0])
        else:
            self.norm_f = None

        if self.norm_f is not None:
            self.layers.append(self.norm_f(config.layer_sizes[0]))

        if config.use_glu:
            self.layers.append(nn.GLU())
        else:
            self.layers.append(self.activation)
        if config.dropout > 0.0:
            self.layers.append(nn.Dropout(config.dropout))

        # Hidden layers
        for i in range(1, len(config.layer_sizes)):
            self.layers.append(
                nn.Linear(config.layer_sizes[i - 1], config.layer_sizes[i])
            )
            if config.batch_norm:
                self.layers.append(nn.BatchNorm1d(config.layer_sizes[i]))
            if config.layer_norm:
                self.layers.append(nn.LayerNorm(config.layer_sizes[i]))
            if config.use_glu:
                self.layers.append(nn.GLU())
            else:
                self.layers.append(self.activation)
            if config.dropout > 0.0:
                self.layers.append(nn.Dropout(config.dropout))

        # Output layer
        self.layers.append(nn.Linear(config.layer_sizes[-1], num_classes))

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
