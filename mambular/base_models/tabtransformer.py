import torch
import torch.nn as nn

from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.mlp_utils import MLPhead
from ..arch_utils.transformer_utils import CustomTransformerEncoderLayer
from ..configs.tabtransformer_config import DefaultTabTransformerConfig
from .basemodel import BaseModel


class TabTransformer(BaseModel):
    """A PyTorch model for tasks utilizing the Transformer architecture and various normalization techniques.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features.
    num_feature_info : dict
        Dictionary containing information about numerical features.
    num_classes : int, optional
        Number of output classes (default is 1).
    config : DefaultFTTransformerConfig, optional
        Configuration object containing default hyperparameters for the model (default is DefaultMambularConfig()).
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    lr : float
        Learning rate.
    lr_patience : int
        Patience for learning rate scheduler.
    weight_decay : float
        Weight decay for optimizer.
    lr_factor : float
        Factor by which the learning rate will be reduced.
    pooling_method : str
        Method to pool the features.
    cat_feature_info : dict
        Dictionary containing information about categorical features.
    num_feature_info : dict
        Dictionary containing information about numerical features.
    embedding_activation : callable
        Activation function for embeddings.
    encoder: callable
        stack of N encoder layers
    norm_f : nn.Module
        Normalization layer.
    num_embeddings : nn.ModuleList
        Module list for numerical feature embeddings.
    cat_embeddings : nn.ModuleList
        Module list for categorical feature embeddings.
    tabular_head : MLPhead
        Multi-layer perceptron head for tabular data.
    cls_token : nn.Parameter
        Class token parameter.
    embedding_norm : nn.Module, optional
        Layer normalization applied after embedding if specified.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultTabTransformerConfig = DefaultTabTransformerConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        if cat_feature_info == {}:
            raise ValueError(
                "You are trying to fit a TabTransformer with no categorical features. \
                    Try using a different model that is better suited for tasks without categorical features."
            )

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

        mlp_input_dim = 0
        for feature_name, input_shape in num_feature_info.items():
            mlp_input_dim += input_shape
        mlp_input_dim += self.hparams.d_model

        self.tabular_head = MLPhead(
            input_dim=mlp_input_dim,
            config=config,
            output_dim=num_classes,
        )

        # pooling
        n_inputs = len(num_feature_info) + len(cat_feature_info)
        self.initialize_pooling_layers(config=config, n_inputs=n_inputs)

    def forward(self, num_features, cat_features):
        """Defines the forward pass of the model.

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
        cat_embeddings = self.embedding_layer(None, cat_features)

        num_features = torch.cat(num_features, dim=1)
        num_embeddings = self.norm_f(num_features)  # type: ignore

        x = self.encoder(cat_embeddings)

        x = self.pool_sequence(x)

        x = torch.cat((x, num_embeddings), axis=1)  # type: ignore
        preds = self.tabular_head(x)

        return preds
