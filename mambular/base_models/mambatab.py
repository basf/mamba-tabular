import torch
import torch.nn as nn
from ..arch_utils.mamba_utils.mamba_arch import Mamba
from ..arch_utils.mlp_utils import MLP
from ..arch_utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
from ..configs.mambatab_config import DefaultMambaTabConfig
from .basemodel import BaseModel


class MambaTab(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultMambaTabConfig = DefaultMambaTabConfig(),
        **kwargs,
    ):
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

        self.initial_layer = nn.Linear(input_dim, config.d_model)
        self.norm_f = LayerNorm(config.d_model)

        self.embedding_activation = self.hparams.get(
            "num_embedding_activation", config.num_embedding_activation
        )

        self.axis = config.axis

        head_activation = self.hparams.get("head_activation", config.head_activation)

        self.tabular_head = MLP(
            self.hparams.get("d_model", config.d_model),
            hidden_units_list=self.hparams.get(
                "head_layer_sizes", config.head_layer_sizes
            ),
            dropout_rate=self.hparams.get("head_dropout", config.head_dropout),
            use_skip_layers=self.hparams.get(
                "head_skip_layers", config.head_skip_layers
            ),
            activation_fn=head_activation,
            use_batch_norm=self.hparams.get(
                "head_use_batch_norm", config.head_use_batch_norm
            ),
            n_output_units=num_classes,
        )

        self.mamba = Mamba(config)

    def forward(self, num_features, cat_features):
        x = num_features + cat_features
        x = torch.cat(x, dim=1)

        x = self.initial_layer(x)
        if self.axis == 1:
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(0)

        x = self.norm_f(x)
        x = self.embedding_activation(x)
        if self.axis == 1:
            x = x.squeeze(1)
        else:
            x = x.squeeze(0)

        preds = self.tabular_head(x)

        return preds
