import torch
import torch.nn as nn
from ..arch_utils.mlp_utils import MLPhead
from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.transformer_utils import CustomTransformerEncoderLayer
from ..configs.fttransformer_config import DefaultFTTransformerConfig
from .basemodel import BaseModel


class FTTransformer(BaseModel):
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
        config: DefaultFTTransformerConfig = DefaultFTTransformerConfig(),
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
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
        encoder_layer = CustomTransformerEncoderLayer(config=config)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.hparams.n_layers,
            norm=self.norm_f,
        )

        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=num_classes,
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

        x = self.pool_sequence(x)

        if self.norm_f is not None:
            x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
