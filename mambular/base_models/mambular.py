import torch
import torch.nn as nn
from ..arch_utils.mamba_arch import Mamba
from ..arch_utils.mlp_utils import MLP
from ..arch_utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
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
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.pooling_method = self.hparams.get("pooling_method", config.pooling_method)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        self.embedding_activation = self.hparams.get(
            "num_embedding_activation", config.num_embedding_activation
        )

        self.mamba = Mamba(
            d_model=self.hparams.get("d_model", config.d_model),
            n_layers=self.hparams.get("n_layers", config.n_layers),
            expand_factor=self.hparams.get("expand_factor", config.expand_factor),
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
            dt_init_floor=self.hparams.get("dt_init_floor", config.dt_init_floor),
            norm=globals()[self.hparams.get("norm", config.norm)],
            activation=self.hparams.get("activation", config.activation),
            bidirectional=self.hparams.get("bidiretional", config.bidirectional),
            use_learnable_interaction=self.hparams.get(
                "use_learnable_interactions", config.use_learnable_interaction
            ),
        )

        norm_layer = self.hparams.get("norm", config.norm)
        if norm_layer == "RMSNorm":
            self.norm_f = RMSNorm(self.hparams.get("d_model", config.d_model))
        elif norm_layer == "LayerNorm":
            self.norm_f = LayerNorm(self.hparams.get("d_model", config.d_model))
        elif norm_layer == "BatchNorm":
            self.norm_f = BatchNorm(self.hparams.get("d_model", config.d_model))
        elif norm_layer == "InstanceNorm":
            self.norm_f = InstanceNorm(self.hparams.get("d_model", config.d_model))
        elif norm_layer == "GroupNorm":
            self.norm_f = GroupNorm(1, self.hparams.get("d_model", config.d_model))
        elif norm_layer == "LearnableLayerScaling":
            self.norm_f = LearnableLayerScaling(
                self.hparams.get("d_model", config.d_model)
            )
        else:
            raise ValueError(f"Unsupported normalization layer: {norm_layer}")

        self.num_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        input_shape,
                        self.hparams.get("d_model", config.d_model),
                        bias=False,
                    ),
                    self.embedding_activation,
                )
                for feature_name, input_shape in num_feature_info.items()
            ]
        )

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_categories + 1, self.hparams.get("d_model", config.d_model)
                )
                for feature_name, num_categories in cat_feature_info.items()
            ]
        )

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

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.hparams.get("d_model", config.d_model))
        )

        if self.hparams.get("layer_norm_after_embedding"):
            self.embedding_norm = nn.LayerNorm(
                self.hparams.get("d_model", config.d_model)
            )

    def __post__init(self):
        pass

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
        batch_size = (
            cat_features[0].size(0) if cat_features != [] else num_features[0].size(0)
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        if len(self.cat_embeddings) > 0 and cat_features:
            cat_embeddings = [
                emb(cat_features[i]) for i, emb in enumerate(self.cat_embeddings)
            ]
            cat_embeddings = torch.stack(cat_embeddings, dim=1)
            cat_embeddings = torch.squeeze(cat_embeddings, dim=2)
            if self.hparams.get("layer_norm_after_embedding"):
                cat_embeddings = self.embedding_norm(cat_embeddings)
        else:
            cat_embeddings = None

        if len(self.num_embeddings) > 0 and num_features:
            num_embeddings = [
                emb(num_features[i]) for i, emb in enumerate(self.num_embeddings)
            ]
            num_embeddings = torch.stack(num_embeddings, dim=1)
            if self.hparams.get("layer_norm_after_embedding"):
                num_embeddings = self.embedding_norm(num_embeddings)
        else:
            num_embeddings = None

        if cat_embeddings is not None and num_embeddings is not None:
            x = torch.cat([cls_tokens, cat_embeddings, num_embeddings], dim=1)
        elif cat_embeddings is not None:
            x = torch.cat([cls_tokens, cat_embeddings], dim=1)
        elif num_embeddings is not None:
            x = torch.cat([cls_tokens, num_embeddings], dim=1)
        else:
            raise ValueError("No features provided to the model.")

        x = self.mamba(x)

        if self.pooling_method == "avg":
            x = torch.mean(x, dim=1)
        elif self.pooling_method == "max":
            x, _ = torch.max(x, dim=1)
        elif self.pooling_method == "sum":
            x = torch.sum(x, dim=1)
        elif self.pooling_method == "cls_token":
            x = x[:, 0]
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")

        x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
