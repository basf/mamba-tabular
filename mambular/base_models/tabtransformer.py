import torch
import torch.nn as nn
from ..arch_utils.mlp_utils import MLP
from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..configs.tabtransformer_config import DefaultTabTransformerConfig
from .basemodel import BaseModel
from ..arch_utils.transformer_utils import CustomTransformerEncoderLayer
from ..arch_utils.layer_utils.normalization_layers import LayerNorm


class TabTransformer(BaseModel):
    """
    A PyTorch model for tasks utilizing the Transformer architecture and various normalization techniques.

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
    tabular_head : MLP
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
        config: DefaultTabTransformerConfig = DefaultTabTransformerConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        if cat_feature_info == {}:
            raise ValueError(
                "You are trying to fit a TabTransformer with no categorical features. Try using a different model that is better suited for tasks without categorical features."
            )

        layer_norm_dim = 0
        for feature_name, input_shape in num_feature_info.items():
            layer_norm_dim += input_shape

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.pooling_method = self.hparams.get("pooling_method", config.pooling_method)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.hparams.get("d_model", config.d_model),
            nhead=self.hparams.get("n_heads", config.n_heads),
            batch_first=True,
            dim_feedforward=self.hparams.get(
                "transformer_dim_feedforward", config.transformer_dim_feedforward
            ),
            dropout=self.hparams.get("attn_dropout", config.attn_dropout),
            activation=self.hparams.get(
                "transformer_activation", config.transformer_activation
            ),
            layer_norm_eps=self.hparams.get("layer_norm_eps", config.layer_norm_eps),
            norm_first=self.hparams.get("norm_first", config.norm_first),
            bias=self.hparams.get("bias", config.bias),
        )

        self.norm_f = get_normalization_layer(config)

        self.norm_embedding = LayerNorm(self.hparams.get("d_model", config.d_model))
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.hparams.get("n_layers", config.n_layers),
            norm=self.norm_embedding,
        )

        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            d_model=self.hparams.get("d_model", config.d_model),
            embedding_activation=self.hparams.get(
                "embedding_activation", config.embedding_activation
            ),
            layer_norm_after_embedding=self.hparams.get(
                "layer_norm_after_embedding", config.layer_norm_after_embedding
            ),
            use_cls=True,
            cls_position=0,
            cat_encoding=self.hparams.get("cat_encoding", config.cat_encoding),
        )

        head_activation = self.hparams.get("head_activation", config.head_activation)

        mlp_input_dim = 0
        for feature_name, input_shape in num_feature_info.items():
            mlp_input_dim += input_shape
        mlp_input_dim += config.d_model
        self.tabular_head = MLP(
            self.hparams.get("d_model", mlp_input_dim),
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

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.hparams.get("d_model", config.d_model))
        )

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
        cat_embeddings = self.embedding_layer(None, cat_features)

        num_features = torch.cat(num_features, dim=1)
        num_embeddings = self.norm_f(num_features)

        x = self.encoder(cat_embeddings)

        if self.pooling_method == "avg":
            x = torch.mean(x, dim=1)
        elif self.pooling_method == "max":
            x, _ = torch.max(x, dim=1)
        elif self.pooling_method == "sum":
            x = torch.sum(x, dim=1)
        elif self.pooling_method == "cls":
            x = x[:, 0]
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")

        x = torch.cat((x, num_embeddings), axis=1)
        preds = self.tabular_head(x)

        return preds
