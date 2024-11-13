import torch
import torch.nn as nn
from ..configs.tabm_config import DefaultTabMConfig
from .basemodel import BaseModel
from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.layer_utils.batch_ensemble_layer import LinearBatchEnsembleLayer
from ..arch_utils.layer_utils.sn_linear import SNLinear


class TabM(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultTabMConfig = DefaultTabMConfig(),
        **kwargs,
    ):
        # Pass config to BaseModel
        super().__init__(config=config, **kwargs)

        # Save hparams including config attributes
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        # Use self.hparams for configuration attributes
        self.layer_sizes = self.hparams.get("layer_sizes", [256, 256])
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.average_ensembles = self.hparams.get("average_ensembles", True)
        self.average_embeddings = self.hparams.get("average_embeddings", False)

        # Initialize layers
        self.layers = nn.ModuleList()
        self.use_glu = self.hparams.get("use_glu", False)
        self.activation = self.hparams.get("activation", nn.SELU())
        self.use_embeddings = self.hparams.get("use_embeddings", True)

        # Embedding layer
        if self.use_embeddings:
            self.embedding_layer = EmbeddingLayer(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
                config=config,
            )

            if self.hparams.get("average_embeddings", config.average_embeddings):
                input_dim = self.hparams.get("d_model", config.d_model)
            else:
                input_dim = (
                    len(num_feature_info) + len(cat_feature_info)
                ) * config.d_model
                print(input_dim)

        else:
            # Calculate input dimension
            input_dim = sum(input_shape for input_shape in num_feature_info.values())
            input_dim += len(cat_feature_info)

        # Input layer with batch ensembling
        self.layers.append(
            LinearBatchEnsembleLayer(
                in_features=input_dim,
                out_features=self.layer_sizes[0],
                ensemble_size=self.hparams.get("ensemble_size", config.ensemble_size),
                ensemble_scaling_in=self.hparams.get(
                    "ensemble_scaling_in", config.ensemble_scaling_in
                ),
                ensemble_scaling_out=self.hparams.get(
                    "ensemble_scaling_out", config.ensemble_scaling_out
                ),
                ensemble_bias=self.hparams.get("ensemble_bias", config.ensemble_bias),
                scaling_init=self.hparams.get("scaling_init", config.scaling_init),
            )
        )
        if self.hparams.get("batch_norm", config.batch_norm):
            self.layers.append(nn.BatchNorm1d(self.layer_sizes[0]))

        self.norm_f = get_normalization_layer(config)
        if self.norm_f is not None:
            self.layers.append(self.norm_f(self.layer_sizes[0]))

        # Optional activation and dropout
        if self.hparams.get("use_glu", config.use_glu):
            self.layers.append(nn.GLU())
        else:
            self.layers.append(self.activation)
        if self.hparams.get("dropout", config.dropout) > 0.0:
            self.layers.append(nn.Dropout(self.hparams.get("dropout", config.dropout)))

        # Hidden layers with batch ensembling
        for i in range(1, len(self.layer_sizes)):
            if self.hparams.get("model_type", config.model_type) == "mini":
                self.layers.append(
                    LinearBatchEnsembleLayer(
                        in_features=self.layer_sizes[i - 1],
                        out_features=self.layer_sizes[i],
                        ensemble_size=self.hparams.get(
                            "ensemble_size", config.ensemble_size
                        ),
                        ensemble_scaling_in=False,
                        ensemble_scaling_out=False,
                        ensemble_bias=self.hparams.get(
                            "ensemble_bias", config.ensemble_bias
                        ),
                        scaling_init="ones",
                    )
                )
            else:
                self.layers.append(
                    LinearBatchEnsembleLayer(
                        in_features=self.layer_sizes[i - 1],
                        out_features=self.layer_sizes[i],
                        ensemble_size=self.hparams.get(
                            "ensemble_size", config.ensemble_size
                        ),
                        ensemble_scaling_in=self.hparams.get(
                            "ensemble_scaling_in", config.ensemble_scaling_in
                        ),
                        ensemble_scaling_out=self.hparams.get(
                            "ensemble_scaling_out", config.ensemble_scaling_out
                        ),
                        ensemble_bias=self.hparams.get(
                            "ensemble_bias", config.ensemble_bias
                        ),
                        scaling_init="ones",
                    )
                )

            if self.hparams.get("use_glu", config.use_glu):
                self.layers.append(nn.GLU())
            else:
                self.layers.append(self.activation)
            if self.hparams.get("dropout", config.dropout) > 0.0:
                self.layers.append(
                    nn.Dropout(self.hparams.get("dropout", config.dropout))
                )

        if self.average_ensembles:
            self.final_layer = nn.Linear(self.layer_sizes[-1], num_classes)
        else:
            self.final_layer = SNLinear(
                self.hparams.get("ensemble_size", config.ensemble_size),
                self.layer_sizes[-1],
                num_classes,
            )

    def forward(self, num_features, cat_features) -> torch.Tensor:
        """
        Forward pass of the TabM model with batch ensembling.

        Parameters
        ----------
        num_features : torch.Tensor
            Numerical features tensor.
        cat_features : torch.Tensor
            Categorical features tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # Handle embeddings if used
        if self.use_embeddings:
            x = self.embedding_layer(num_features, cat_features)
            # Option 1: Average over feature dimension (N)
            if self.average_embeddings:
                x = x.mean(dim=1)  # Shape: (B, D)
            # Option 2: Flatten feature and embedding dimensions
            else:
                B, N, D = x.shape
                x = x.reshape(B, N * D)  # Shape: (B, N * D)

        else:
            x = num_features + cat_features
            x = torch.cat(x, dim=1)

        # Process through layers with optional skip connections
        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], LinearBatchEnsembleLayer):
                out = self.layers[i](x)
                # `out` shape is expected to be (batch_size, ensemble_size, out_features)
                if self.skip_connections and x.shape == out.shape:
                    x = x + out
                else:
                    x = out
            else:
                x = self.layers[i](x)

        # Final ensemble output from the last ConfigurableBatchEnsembleLayer
        x = self.layers[-1](x)  # Shape (batch_size, ensemble_size, num_classes)

        # Option 1: Averaging across ensemble outputs
        if self.average_ensembles:
            x = x.mean(dim=1)  # Shape (batch_size, num_classes)

        x = self.final_layer(x)  # Shape (batch_size, num_classes)

        print(x.shape)
        return x
