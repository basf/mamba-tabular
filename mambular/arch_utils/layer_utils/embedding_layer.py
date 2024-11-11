import torch
import torch.nn as nn
from .embedding_tree import NeuralEmbeddingTree


class EmbeddingLayer(nn.Module):
    def __init__(self, num_feature_info, cat_feature_info, config):
        """
        Embedding layer that handles numerical and categorical embeddings.

        Parameters
        ----------
        num_feature_info : dict
            Dictionary where keys are numerical feature names and values are their respective input dimensions.
        cat_feature_info : dict
            Dictionary where keys are categorical feature names and values are the number of categories for each feature.
        config : Config
            Configuration object containing all required settings.
        """
        super(EmbeddingLayer, self).__init__()

        self.d_model = config.d_model
        self.embedding_activation = getattr(
            config, "embedding_activation", nn.Identity()
        )
        self.layer_norm_after_embedding = getattr(
            config, "layer_norm_after_embedding", False
        )
        self.use_cls = getattr(config, "use_cls", False)
        self.cls_position = getattr(config, "cls_position", 0)
        self.cat_encoding = getattr(config, "cat_encoding", "int")
        self.embedding_dropout = (
            nn.Dropout(getattr(config, "embedding_dropout", 0.0))
            if getattr(config, "embedding_dropout", None) is not None
            else None
        )

        self.num_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_shape, self.d_model, bias=False),
                    self.embedding_activation,
                )
                for feature_name, input_shape in num_feature_info.items()
            ]
        )

        # Initialize categorical embeddings
        self.cat_embeddings = nn.ModuleList()
        for feature_name, num_categories in cat_feature_info.items():
            if self.cat_encoding == "int":
                self.cat_embeddings.append(
                    nn.Sequential(
                        nn.Embedding(num_categories + 1, self.d_model),
                        self.embedding_activation,
                    )
                )
            elif self.cat_encoding == "one-hot":
                self.cat_embeddings.append(
                    nn.Sequential(
                        OneHotEncoding(num_categories),
                        nn.Linear(num_categories, self.d_model, bias=False),
                        self.embedding_activation,
                    )
                )

        # Class token if required
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Layer normalization if required
        if self.layer_norm_after_embedding:
            self.embedding_norm = nn.LayerNorm(self.d_model)

        # Sequence length
        self.seq_len = len(self.num_embeddings) + len(self.cat_embeddings)

    def forward(self, num_features=None, cat_features=None):
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        num_features : Tensor, optional
            Tensor containing the numerical features.
        cat_features : Tensor, optional
            Tensor containing the categorical features.

        Returns
        -------
        Tensor
            The output embeddings of the model.

        Raises
        ------
        ValueError
            If no features are provided to the model.
        """
        # Class token initialization
        if self.use_cls:
            batch_size = (
                cat_features[0].size(0)
                if cat_features != []
                else num_features[0].size(0)
            )
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Process categorical embeddings
        if self.cat_embeddings and cat_features is not None:
            cat_embeddings = [
                emb(cat_features[i]) for i, emb in enumerate(self.cat_embeddings)
            ]
            cat_embeddings = torch.stack(cat_embeddings, dim=1)
            cat_embeddings = torch.squeeze(cat_embeddings, dim=2)
            if self.layer_norm_after_embedding:
                cat_embeddings = self.embedding_norm(cat_embeddings)
        else:
            cat_embeddings = None

        # Process numerical embeddings
        if self.num_embeddings and num_features is not None:
            num_embeddings = [
                emb(num_features[i]) for i, emb in enumerate(self.num_embeddings)
            ]
            num_embeddings = torch.stack(num_embeddings, dim=1)
            if self.layer_norm_after_embedding:
                num_embeddings = self.embedding_norm(num_embeddings)
        else:
            num_embeddings = None

        # Combine categorical and numerical embeddings
        if cat_embeddings is not None and num_embeddings is not None:
            x = torch.cat([cat_embeddings, num_embeddings], dim=1)
        elif cat_embeddings is not None:
            x = cat_embeddings
        elif num_embeddings is not None:
            x = num_embeddings
        else:
            raise ValueError("No features provided to the model.")

        # Add class token if required
        if self.use_cls:
            if self.cls_position == 0:
                x = torch.cat([cls_tokens, x], dim=1)
            elif self.cls_position == 1:
                x = torch.cat([x, cls_tokens], dim=1)
            else:
                raise ValueError(
                    "Invalid cls_position value. It should be either 0 or 1."
                )

        # Apply dropout to embeddings if specified in config
        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        return x


class OneHotEncoding(nn.Module):
    def __init__(self, num_categories):
        super(OneHotEncoding, self).__init__()
        self.num_categories = num_categories

    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=self.num_categories).float()
