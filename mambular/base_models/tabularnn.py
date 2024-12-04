import torch
import torch.nn as nn
from ..arch_utils.mlp_utils import MLPhead
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
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.returns_ensemble = False
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        self.rnn = ConvRNN(config)

        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config,
        )

        self.tabular_head = MLPhead(
            input_dim=self.hparams.dim_feedforward,
            config=config,
            output_dim=num_classes,
        )

        self.linear = nn.Linear(
            self.hparams.d_model,
            self.hparams.dim_feedforward,
        )

        temp_config = replace(config, d_model=config.dim_feedforward)
        self.norm_f = get_normalization_layer(temp_config)

        # pooling
        n_inputs = len(num_feature_info) + len(cat_feature_info)
        self.initialize_pooling_layers(config=config, n_inputs=n_inputs)

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

        x = self.pool_sequence(out)
        x = x + z
        if self.norm_f is not None:
            x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
