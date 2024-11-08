import torch
import torch.nn as nn
from ..arch_utils.mlp_utils import MLP
from ..configs.tabularnn_config import DefaultTabulaRNNConfig
from .basemodel import BaseModel
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.rnn_utils import ConvRNN
from ..arch_utils.get_norm_fn import get_normalization_layer
from dataclasses import replace


class TabulaRNN(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultTabulaRNNConfig = DefaultTabulaRNNConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.pooling_method = self.hparams.get("pooling_method", config.pooling_method)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        self.rnn = ConvRNN(
            model_type=self.hparams.get("model_type", config.model_type),
            input_size=self.hparams.get("d_model", config.d_model),
            hidden_size=self.hparams.get("dim_feedforward", config.dim_feedforward),
            num_layers=self.hparams.get("n_layers", config.n_layers),
            bidirectional=self.hparams.get("bidirectional", config.bidirectional),
            rnn_dropout=self.hparams.get("rnn_dropout", config.rnn_dropout),
            bias=self.hparams.get("bias", config.bias),
            conv_bias=self.hparams.get("conv_bias", config.conv_bias),
            rnn_activation=self.hparams.get("rnn_activation", config.rnn_activation),
            d_conv=self.hparams.get("d_conv", config.d_conv),
            residuals=self.hparams.get("residuals", config.residuals),
        )

        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config,
        )

        head_activation = self.hparams.get("head_activation", config.head_activation)

        self.tabular_head = MLP(
            self.hparams.get("dim_feedforward", config.dim_feedforward),
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

        self.linear = nn.Linear(
            self.hparams.get("d_model", config.d_model),
            self.hparams.get("dim_feedforward", config.dim_feedforward),
        )

        temp_config = replace(config, d_model=config.dim_feedforward)
        self.norm_f = get_normalization_layer(temp_config)

    def forward(self, num_features, cat_features):
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        num_features : Tensor
            Tensor containing the numerical features.
        cat_features : Tensor
            Tensor containing the categorical features.

        Returns
        -------
        Tensor
            The output predictions of the model.
        """

        x = self.embedding_layer(num_features, cat_features)
        # RNN forward pass
        out, _ = self.rnn(x)
        z = self.linear(torch.mean(x, dim=1))

        if self.pooling_method == "avg":
            x = torch.mean(out, dim=1)
        elif self.pooling_method == "max":
            x, _ = torch.max(out, dim=1)
        elif self.pooling_method == "sum":
            x = torch.sum(out, dim=1)
        elif self.pooling_method == "last":
            x = x[:, -1, :]
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")
        x = x + z
        if self.norm_f is not None:
            x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
