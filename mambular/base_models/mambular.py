import torch
import torch.nn as nn

from ..arch_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.mamba_arch import Mamba
from ..arch_utils.mlp_utils import MLP
from ..arch_utils.normalization_layers import (BatchNorm, GroupNorm,
                                               InstanceNorm, LayerNorm,
                                               LearnableLayerScaling, RMSNorm)
from ..configs.mambular_config import DefaultMambularConfig
from .basemodel import BaseModel


class Mambular(BaseModel):
    """
    A PyTorch model for tasks utilizing the Mamba architecture and various normalization techniques.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features.
    num_feature_info : dict
        Dictionary containing information about numerical features.
    num_classes : int, optional
        Number of output classes (default is 1).
    config : DefaultMambularConfig, optional
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
    mamba : Mamba
        Mamba architecture component.
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
        config: DefaultMambularConfig = DefaultMambularConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(
            ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get(
            "weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.pooling_method = self.hparams.get(
            "pooling_method", config.pooling_method)
        self.shuffle_embeddings = self.hparams.get(
            "shuffle_embeddings", config.shuffle_embeddings
        )
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        self.mamba = Mamba(
            d_model=self.hparams.get("d_model", config.d_model),
            n_layers=self.hparams.get("n_layers", config.n_layers),
            expand_factor=self.hparams.get(
                "expand_factor", config.expand_factor),
            bias=self.hparams.get("bias", config.bias),
            d_conv=self.hparams.get("d_conv", config.d_conv),
            conv_bias=self.hparams.get("conv_bias", config.conv_bias),
            dropout=self.hparams.get("dropout", config.dropout),
            dt_rank=self.hparams.get("dt_rank", config.dt_rank),
            d_state=self.hparams.get("d_state", config.d_state),
            dt_scale=self.hparams.get("dt_scale", config.dt_scale),
            dt_init=self.hparams.get("dt_init", config.dt_init),
            dt_max=self.hparams.get("dt_max", config.dt_max),
            dt_min=self.hparams.get("dt_min", config.dt_min),
            dt_init_floor=self.hparams.get(
                "dt_init_floor", config.dt_init_floor),
            norm=globals()[self.hparams.get("norm", config.norm)],
            activation=self.hparams.get("activation", config.activation),
            bidirectional=self.hparams.get(
                "bidiretional", config.bidirectional),
            use_learnable_interaction=self.hparams.get(
                "use_learnable_interactions", config.use_learnable_interaction
            ),
            AD_weight_decay=self.hparams.get(
                "AB_weight_decay", config.AD_weight_decay),
            BC_layer_norm=self.hparams.get(
                "AB_layer_norm", config.BC_layer_norm),
            layer_norm_eps=self.hparams.get(
                "layer_norm_eps", config.layer_norm_eps),
        )
        norm_layer = self.hparams.get("norm", config.norm)
        if norm_layer == "RMSNorm":
            self.norm_f = RMSNorm(
                self.hparams.get("d_model", config.d_model), eps=config.layer_norm_eps
            )
        elif norm_layer == "LayerNorm":
            self.norm_f = LayerNorm(
                self.hparams.get("d_model", config.d_model), eps=config.layer_norm_eps
            )
        elif norm_layer == "BatchNorm":
            self.norm_f = BatchNorm(
                self.hparams.get("d_model", config.d_model), eps=config.layer_norm_eps
            )
        elif norm_layer == "InstanceNorm":
            self.norm_f = InstanceNorm(
                self.hparams.get("d_model", config.d_model), eps=config.layer_norm_eps
            )
        elif norm_layer == "GroupNorm":
            self.norm_f = GroupNorm(
                1,
                self.hparams.get("d_model", config.d_model),
                eps=config.layer_norm_eps,
            )
        elif norm_layer == "LearnableLayerScaling":
            self.norm_f = LearnableLayerScaling(
                self.hparams.get("d_model", config.d_model)
            )
        else:
            raise ValueError(f"Unsupported normalization layer: {norm_layer}")

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
            use_cls=False,
            cls_position=-1,
            cat_encoding=self.hparams.get("cat_encoding", config.cat_encoding),
        )

        head_activation = self.hparams.get(
            "head_activation", config.head_activation)

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

        if self.pooling_method == "cls":
            self.use_cls = True
        else:
            self.use_cls = self.hparams.get("use_cls", config.use_cls)

        if self.shuffle_embeddings:
            self.perm = torch.randperm(self.embedding_layer.seq_len)

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

        if self.shuffle_embeddings:
            x = x[:, self.perm, :]

        x = self.mamba(x)

        if self.pooling_method == "avg":
            x = torch.mean(x, dim=1)
        elif self.pooling_method == "max":
            x, _ = torch.max(x, dim=1)
        elif self.pooling_method == "sum":
            x = torch.sum(x, dim=1)
        elif self.pooling_method == "cls":
            x = x[:, -1]
        elif self.pooling_method == "last":
            x = x[:, -1]
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")

        x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
