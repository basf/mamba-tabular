import torch
import torch.nn as nn
from typing import Any
from ..configs.resnet_config import DefaultResNetConfig
from .basemodel import BaseModel
from ..arch_utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
from ..arch_utils.resnet_utils import ResidualBlock


class ResNet(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultResNetConfig = DefaultResNetConfig(),
        **kwargs,
    ):
        """
        ResNet model for structured data.

        Parameters
        ----------
        cat_feature_info : Any
            Information about categorical features.
        num_feature_info : Any
            Information about numerical features.
        num_classes : int, optional
            Number of output classes, by default 1.
        config : DefaultResNetConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultResNetConfig().
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

        self.activation = config.activation
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

        self.initial_layer = nn.Linear(input_dim, config.layer_sizes[0])

        self.blocks = nn.ModuleList()
        for i in range(config.num_blocks):
            input_dim = config.layer_sizes[i]
            output_dim = (
                config.layer_sizes[i + 1]
                if i + 1 < len(config.layer_sizes)
                else config.layer_sizes[-1]
            )
            block = ResidualBlock(
                input_dim,
                output_dim,
                self.activation,
                self.norm_f,
                config.dropout,
            )
            self.blocks.append(block)

        self.output_layer = nn.Linear(config.layer_sizes[-1], num_classes)

    def forward(self, num_features, cat_features):
        """
        Forward pass of the ResNet model.

        Parameters
        ----------
        num_features : torch.Tensor
            Tensor of numerical features.
        cat_features : torch.Tensor, optional
            Tensor of categorical features.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = num_features + cat_features
        x = torch.cat(x, dim=1)

        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x
