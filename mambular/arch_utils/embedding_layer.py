import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        num_feature_info,
        cat_feature_info,
        d_model,
        embedding_activation=nn.Identity(),
        layer_norm_after_embedding=False,
        use_cls=False,
        cls_position=0,
    ):
        """
        Embedding layer that handles numerical and categorical embeddings.

        Parameters
        ----------
        num_feature_info : dict
            Dictionary where keys are numerical feature names and values are their respective input dimensions.
        cat_feature_info : dict
            Dictionary where keys are categorical feature names and values are the number of categories for each feature.
        d_model : int
            Dimensionality of the embeddings.
        embedding_activation : nn.Module, optional
            Activation function to apply after embedding. Default is `nn.Identity()`.
        layer_norm_after_embedding : bool, optional
            If True, applies layer normalization after embeddings. Default is `False`.
        use_cls : bool, optional
            If True, includes a class token in the embeddings. Default is `False`.
        cls_position : int, optional
            Position to place the class token, either at the start (0) or end (1) of the sequence. Default is `0`.

        Methods
        -------
        forward(num_features=None, cat_features=None)
            Defines the forward pass of the model.
        """
        super(EmbeddingLayer, self).__init__()

        self.d_model = d_model
        self.embedding_activation = embedding_activation
        self.layer_norm_after_embedding = layer_norm_after_embedding
        self.use_cls = use_cls
        self.cls_position = cls_position

        self.num_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_shape, d_model, bias=False),
                    self.embedding_activation,
                )
                for feature_name, input_shape in num_feature_info.items()
            ]
        )

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(num_categories + 1, d_model),
                    self.embedding_activation,
                )
                for feature_name, num_categories in cat_feature_info.items()
            ]
        )

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if layer_norm_after_embedding:
            self.embedding_norm = nn.LayerNorm(d_model)

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
        if self.use_cls:
            batch_size = (
                cat_features[0].size(0)
                if cat_features != []
                else num_features[0].size(0)
            )
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)

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

        if self.num_embeddings and num_features is not None:
            num_embeddings = [
                emb(num_features[i]) for i, emb in enumerate(self.num_embeddings)
            ]
            num_embeddings = torch.stack(num_embeddings, dim=1)
            if self.layer_norm_after_embedding:
                num_embeddings = self.embedding_norm(num_embeddings)
        else:
            num_embeddings = None

        if cat_embeddings is not None and num_embeddings is not None:
            x = torch.cat([cat_embeddings, num_embeddings], dim=1)
        elif cat_embeddings is not None:
            x = cat_embeddings
        elif num_embeddings is not None:
            x = num_embeddings
        else:
            raise ValueError("No features provided to the model.")

        if self.use_cls:
            if self.cls_position == 0:
                x = torch.cat([cls_tokens, x], dim=1)
            elif self.cls_position == 1:
                x = torch.cat([x, cls_tokens], dim=1)
            else:
                raise ValueError(
                    "Invalid cls_position value. It should be either 0 or 1."
                )

        return x
