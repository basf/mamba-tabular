import torch
import torch.nn as nn
from ..configs.cnn_config import DefaultCNNConfig
from .basemodel import BaseModel
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.cnn_utils import CNNBlock


class CNN(BaseModel):
    """
    A convolutional neural network (CNN) model designed for tabular data with support for categorical
    and numerical features, configurable embeddings, and dynamic flattened size computation.

    Attributes
    ----------
    embedding_layer : EmbeddingLayer
        A layer that generates embeddings for categorical and numerical features.
    cnn : CNNBlock
        A modular CNN block for feature extraction.
    fc : nn.Sequential
        A fully connected layer for final predictions.

    Methods
    -------
    forward(num_features, cat_features):
        Forward pass through the embedding, CNN, and fully connected layers.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultCNNConfig = DefaultCNNConfig(),
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=[])

        self.returns_ensemble = False
        self.n_features = len(num_feature_info) + len(cat_feature_info)

        # Initialize the embedding layer
        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config,
        )

        # CNN block
        self.cnn = CNNBlock(config)
        n_features = len(num_feature_info) + len(cat_feature_info)

        # Dynamically compute flattened size
        with torch.no_grad():
            sample_input = torch.zeros(
                1,
                config.input_channels,
                n_features,
                config.d_model,
            )
            sample_output = self.cnn(sample_input)
            flattened_size = sample_output.view(1, -1).size(1)
            print(flattened_size)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, num_classes),
        )

    def forward(self, num_features, cat_features):
        x = self.embedding_layer(num_features, cat_features)
        x = x.unsqueeze(1)
        # Generate embeddings (x) with shape (N, J, D)

        x = self.cnn(x)
        preds = self.fc(x)
        return preds
