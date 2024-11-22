import torch
import torch.nn as nn
from ..arch_utils.layer_utils.sn_linear import SNLinear
from ..configs.batchtabrnn_config import DefaultBatchTabRNNConfig
from .basemodel import BaseModel
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.rnn_utils import EnsembleConvRNN
from ..arch_utils.get_norm_fn import get_normalization_layer
from dataclasses import replace
from ..arch_utils.layer_utils.sn_linear import SNLinear


class BatchTabRNN(BaseModel):
    """
    A batch ensemble model combining RNN and tabular data handling for multivariate time series or sequential tabular data.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultBatchTabRNNConfig, optional
        Configuration object containing model hyperparameters such as dropout rates, hidden layer sizes, ensemble settings,
        and other architectural configurations, by default DefaultBatchTabRNNConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    pooling_method : str
        The pooling method to aggregate sequence or ensemble features, specified in config.
    ensemble_first : bool
        Flag indicating if ensembles should be processed before pooling over the sequence.
    embedding_layer : EmbeddingLayer
        Layer for embedding categorical and numerical features.
    rnn : EnsembleConvRNN
        Ensemble RNN layer for processing sequential data.
    tabular_head : MLPhead
        MLPhead layer to produce the final prediction based on the output of the RNN and pooling layers.
    linear : nn.Linear
        Linear transformation layer for projecting features into a different dimension.
    norm_f : nn.Module
        Normalization layer.
    ensemble_linear : nn.Linear, optional
        Linear layer to learn a weighted combination of ensemble outputs, if configured.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including embedding, RNN, pooling, and prediction steps.

    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultBatchTabRNNConfig = DefaultBatchTabRNNConfig(),
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        if not self.hparams.average_ensembles:
            self.returns_ensemble = True  # Directly set ensemble flag
        else:
            self.returns_ensemble = False

        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config,
        )
        self.rnn = EnsembleConvRNN(config=config)

        self.linear = nn.Linear(
            self.hparams.d_model,
            self.hparams.dim_feedforward,
        )

        temp_config = replace(config, d_model=config.dim_feedforward)
        self.norm_f = get_normalization_layer(temp_config)

        if self.hparams.average_ensembles:
            self.final_layer = nn.Linear(self.hparams.dim_feedforward, num_classes)
        else:
            self.final_layer = SNLinear(
                self.hparams.ensemble_size,
                self.hparams.dim_feedforward,
                num_classes,
            )

        n_inputs = len(num_feature_info) + len(cat_feature_info)
        self.initialize_pooling_layers(config=config, n_inputs=n_inputs)

    def forward(self, num_features, cat_features):
        x = self.embedding_layer(num_features, cat_features)

        # RNN forward pass
        out, _ = self.rnn(
            x
        )  # Shape: (batch_size, sequence_length, ensemble_size, hidden_size)

        out = self.pool_sequence(out)  # Shape: (batch_size, ensemble_size, hidden_size)

        if self.hparams.average_ensembles:
            x = out.mean(axis=1)  # Shape (batch_size, num_classes)

        x = self.final_layer(
            out
        )  # Shape (batch_size, (ensemble_size), num_classes) if not averaged

        if not self.hparams.average_ensembles:
            x = x.squeeze(-1)

        return x
