import torch
import torch.nn as nn
from ..arch_utils.mlp_utils import MLP

from ..configs.rnn_config import DefaultRNNConfig
from .basemodel import BaseModel


import torch
import torch.nn as nn
import torch.optim as optim


class RNN(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultRNNConfig = DefaultRNNConfig(),
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

        self.rnn = nn.RNN(
            input_size=self.hparams.get("d_model", config.d_model),
            hidden_size=self.hparams.get("dim_feedforward", config.dim_feedforward),
            num_layers=self.hparams.get("n_layers", config.n_layers),
            bidirectional=self.hparams.get("bidirectional", config.bidirectional),
            batch_first=True,
            dropout=self.hparams.get("rnn_dropout", config.rnn_dropout),
            bias=self.hparams.get("bias", config.bias),
            nonlinearity=self.hparams.get("rnn_activation", config.rnn_activation),
        )

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

        if self.hparams.get("layer_norm_after_embedding"):
            self.embedding_norm = nn.LayerNorm(
                self.hparams.get("d_model", config.d_model)
            )

        self.linear = nn.Linear(config.d_model, config.dim_feedforward)

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
            x = torch.cat([cat_embeddings, num_embeddings], dim=1)
        elif cat_embeddings is not None:
            x = torch.cat([cat_embeddings], dim=1)
        elif num_embeddings is not None:
            x = torch.cat([num_embeddings], dim=1)
        else:
            raise ValueError("No features provided to the model.")

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
        preds = self.tabular_head(x)

        return preds
