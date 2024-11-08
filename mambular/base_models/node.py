from .basemodel import BaseModel
from ..configs.node_config import DefaultNODEConfig
import torch
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.node_utils import DenseBlock
from ..arch_utils.mlp_utils import MLP


class NODE(BaseModel):
    """
    Neural Oblivious Decision Ensemble (NODE) Model. Slightly different with a MLP as a tabular task specific head.

    NODE is a neural decision tree model that processes both categorical and numerical features.
    This class combines embedding layers, a dense decision tree block, and an MLP head for tabular
    data prediction tasks.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary mapping categorical feature names to their input shapes.
    num_feature_info : dict
        Dictionary mapping numerical feature names to their input shapes.
    num_classes : int, optional
        Number of output classes. Default is 1.
    config : DefaultNODEConfig, optional
        Configuration object that holds model hyperparameters. Default is `DefaultNODEConfig`.
    **kwargs : dict
        Additional arguments for the base model.

    Attributes
    ----------
    lr : float
        Learning rate for the optimizer.
    lr_patience : int
        Number of epochs without improvement before reducing the learning rate.
    weight_decay : float
        Weight decay factor for regularization.
    lr_factor : float
        Factor by which to reduce the learning rate.
    cat_feature_info : dict
        Information about categorical features.
    num_feature_info : dict
        Information about numerical features.
    use_embeddings : bool
        Whether to use embeddings for categorical and numerical features.
    embedding_layer : EmbeddingLayer, optional
        Embedding layer for feature transformation.
    d_out : int
        Output dimensionality.
    block : DenseBlock
        DenseBlock layer that implements the decision tree ensemble.
    tabular_head : MLP
        MLP layer that serves as the output head of the model.

    Methods
    -------
    forward(num_features, cat_features)
        Performs the forward pass, processing numerical and categorical features to produce predictions.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNODEConfig = DefaultNODEConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.use_embeddings = self.hparams.get("use_embeddings", config.use_embeddings)

        input_dim = 0
        for feature_name, input_shape in num_feature_info.items():
            input_dim += input_shape
        for feature_name, input_shape in cat_feature_info.items():
            input_dim += 1

        if self.use_embeddings:
            input_dim = (
                len(num_feature_info) * config.d_model
                + len(cat_feature_info) * config.d_model
            )

            self.embedding_layer = EmbeddingLayer(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
                config=config,
            )

        self.d_out = num_classes
        self.block = DenseBlock(
            input_dim=input_dim,
            num_layers=config.num_layers,
            layer_dim=config.layer_dim,
            depth=config.depth,
            tree_dim=config.tree_dim,
            flatten_output=True,
        )

        head_activation = self.hparams.get("head_activation", config.head_activation)

        self.tabular_head = MLP(
            config.num_layers * config.layer_dim,
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

    def forward(self, num_features, cat_features):
        """
        Forward pass through the NODE model.

        Parameters
        ----------
        num_features : torch.Tensor
            Numerical features tensor of shape [batch_size, num_numerical_features].
        cat_features : torch.Tensor
            Categorical features tensor of shape [batch_size, num_categorical_features].

        Returns
        -------
        torch.Tensor
            Model output of shape [batch_size, num_classes].
        """
        if self.use_embeddings:
            x = self.embedding_layer(num_features, cat_features)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
        else:
            x = num_features + cat_features
            x = torch.cat(x, dim=1)

        x = self.block(x).squeeze(-1)
        x = self.tabular_head(x)
        return x
