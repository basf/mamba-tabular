import torch
import torch.nn as nn
from ..arch_utils.mlp_utils import MLPhead
from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.transformer_utils import BatchEnsembleTransformerEncoder
from ..configs.ftet_config import DefaultFTETConfig
from .basemodel import BaseModel
from ..arch_utils.layer_utils.sn_linear import SNLinear


class FTET(BaseModel):
    """
    A Feature Transformer model for tabular data with categorical and numerical features, using embedding, transformer
    encoding, and pooling to produce final predictions.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultFTTransformerConfig, optional
        Configuration object containing model hyperparameters such as dropout rates, hidden layer sizes,
        transformer settings, and other architectural configurations, by default DefaultFTTransformerConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    pooling_method : str
        The pooling method to aggregate features after transformer encoding.
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    embedding_layer : EmbeddingLayer
        Layer for embedding categorical and numerical features.
    norm_f : nn.Module
        Normalization layer for the transformer output.
    encoder : nn.TransformerEncoder
        Transformer encoder for sequential processing of embedded features.
    tabular_head : MLPhead
        MLPhead layer to produce the final prediction based on the output of the transformer encoder.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including embedding, transformer encoding, pooling, and prediction steps.

    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultFTETConfig = DefaultFTETConfig(),
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

        # embedding layer
        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config,
        )

        # transformer encoder
        self.norm_f = get_normalization_layer(config)
        self.encoder = BatchEnsembleTransformerEncoder(config)

        if self.hparams.average_ensembles:
            self.final_layer = nn.Linear(self.hparams.d_model, num_classes)
        else:
            self.final_layer = SNLinear(
                self.hparams.ensemble_size,
                self.hparams.d_model,
                num_classes,
            )

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

        x = self.encoder(x)

        x = self.pool_sequence(x)  # Shape: (batch_size, ensemble_size, hidden_size)

        if self.hparams.average_ensembles:
            x = x.mean(axis=1)  # Shape (batch_size, num_classes)

        x = self.final_layer(
            x
        )  # Shape (batch_size, (ensemble_size), num_classes) if not averaged

        if not self.hparams.average_ensembles:
            x = x.squeeze(-1)

        return x
