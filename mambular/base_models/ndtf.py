import torch
import torch.nn as nn
from ..configs.ndtf_config import DefaultNDTFConfig
from .basemodel import BaseModel
from ..arch_utils.neural_decision_tree import NeuralDecisionTree
import numpy as np


class NDTF(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNDTFConfig = DefaultNDTFConfig(),
        **kwargs,
    ):
        """
        Initializes the NDTF model with the given configuration.

        Parameters
        ----------
        cat_feature_info : Any
            Information about categorical features.
        num_feature_info : Any
            Information about numerical features.

        num_classes : int, optional
            Number of output classes, by default 1.
        config : DefaultNDTFConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultNDTFConfig().
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.penalty_factor = config.penalty_factor

        input_dim = 0
        for feature_name, input_shape in num_feature_info.items():
            input_dim += input_shape
        for feature_name, input_shape in cat_feature_info.items():
            input_dim += 1

        self.input_dimensions = [input_dim]

        for _ in range(config.n_ensembles - 1):
            self.input_dimensions.append(np.random.randint(1, input_dim))

        self.trees = nn.ModuleList(
            [
                NeuralDecisionTree(
                    input_dim=self.input_dimensions[idx],
                    depth=np.random.randint(config.min_depth, config.max_depth),
                    output_dim=num_classes,
                    lamda=config.lamda,
                    temperature=config.temperature + np.abs(np.random.normal(0, 0.1)),
                    node_sampling=config.node_sampling,
                )
                for idx in range(config.n_ensembles)
            ]
        )

        self.conv_layer = nn.Conv1d(
            in_channels=self.input_dimensions[0],
            out_channels=1,  # Single channel output if one feature interaction is desired
            kernel_size=self.input_dimensions[0],  # Choose appropriate kernel size
            padding=self.input_dimensions[0]
            - 1,  # To keep output size the same as input_dim if desired
            bias=True,
        )

        self.tree_weights = nn.Parameter(
            torch.full((config.n_ensembles, 1), 1.0 / config.n_ensembles),
            requires_grad=True,
        )

    def forward(self, num_features, cat_features) -> torch.Tensor:
        """
        Forward pass of the NDTF model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = num_features + cat_features
        x = torch.cat(x, dim=1)
        x = self.conv_layer(x.unsqueeze(2))
        x = x.transpose(1, 2).squeeze(-1)

        preds = []

        for idx, tree in enumerate(self.trees):
            tree_input = x[:, : self.input_dimensions[idx]]
            preds.append(tree(tree_input, return_penalty=False))

        preds = torch.stack(preds, dim=1).squeeze(-1)

        return preds @ self.tree_weights

    def penalty_forward(self, num_features, cat_features) -> torch.Tensor:
        """
        Forward pass of the NDTF model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = num_features + cat_features
        x = torch.cat(x, dim=1)
        x = self.conv_layer(x.unsqueeze(2))
        x = x.transpose(1, 2).squeeze(-1)

        penalty = 0.0
        preds = []

        # Iterate over trees and collect predictions and penalties
        for idx, tree in enumerate(self.trees):
            # Select subset of features for the current tree
            tree_input = x[:, : self.input_dimensions[idx]]

            # Get prediction and penalty from the current tree
            pred, pen = tree(tree_input, return_penalty=True)
            preds.append(pred)
            penalty += pen

        # Stack predictions and calculate mean across trees
        preds = torch.stack(preds, dim=1).squeeze(-1)
        return preds @ self.tree_weights, self.penalty_factor * penalty
