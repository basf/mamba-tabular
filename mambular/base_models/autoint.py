import torch
import torch.nn as nn
import torch.nn.functional as F
from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.mlp_utils import MLPhead
from ..configs.autoint_config import DefaultAutoIntConfig
from .basemodel import BaseModel
import numpy as np


class AutoInt(BaseModel):

    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes=1,
        config: DefaultAutoIntConfig = DefaultAutoIntConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])
        self.returns_ensemble = False

        # Embedding Layer
        self.embedding_layer = EmbeddingLayer(
            *feature_information,
            config=config,
        )

        self.norm_f = get_normalization_layer(config)

        # Key-Value Compression (Optional)
        def make_kv_compression():
            assert config.kv_compression
            compression = nn.Linear(
                self.embedding_layer.seq_len,
                int(self.embedding_layer.seq_len * config.kv_compression),
                bias=False,
            )
            nn.init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if config.kv_compression and config.kv_compression_sharing == "layerwise"
            else None
        )

        # Transformer Layers
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.n_layers):
            layer = nn.ModuleDict(
                {
                    "attention": nn.MultiheadAttention(
                        embed_dim=config.d_model,
                        num_heads=config.n_heads,
                        dropout=config.attention_dropout,
                        batch_first=True,
                    ),
                    "linear": nn.Linear(config.d_model, config.d_model, bias=False),
                }
            )
            if not config.prenormalization or layer_idx:
                layer["norm0"] = nn.LayerNorm(config.d_model)

            if config.kv_compression and self.shared_kv_compression is None:
                layer["key_compression"] = make_kv_compression()
                if config.kv_compression_sharing == "headwise":
                    layer["value_compression"] = make_kv_compression()
                else:
                    assert config.kv_compression_sharing == "key-value"

            self.layers.append(layer)

        # Final Layers
        self.activation = F.relu  # AutoInt enforces ReLU
        self.prenormalization = config.prenormalization
        self.residual_dropout = config.residual_dropout
        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=num_classes,
        )

    def _get_kv_compressions(self, layer):
        """Retrieve key and value compression modules."""
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (
                (layer.get("key_compression"), layer.get("value_compression"))
                if "key_compression" in layer and "value_compression" in layer
                else (
                    (layer.get("key_compression"), layer.get("key_compression"))
                    if "key_compression" in layer
                    else (None, None)
                )
            )
        )

    def _start_residual(self, x, layer, norm_idx):
        """Applies prenormalization if enabled."""
        x_residual = x
        if self.prenormalization:
            norm_key = f"norm{norm_idx}"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        """Applies post-normalization and dropout if needed."""
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"norm{norm_idx}"](x)
        return x

    def forward(self, *data):
        x = self.embedding_layer(*data)
        x = self.norm_f(x)  # Apply feature normalization

        for layer in self.layers:
            layer = dict(layer)  # Ensure correct type handling

            x_residual = self._start_residual(x, layer, 0)
            x_residual, _ = layer["attention"](
                x_residual, x_residual, x_residual, need_weights=False
            )

            x = layer["linear"](x)
            x = self._end_residual(x, x_residual, layer, 0)
            x = self.activation(x)

        x = self.pool_sequence(x)

        if self.norm_f is not None:
            x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
