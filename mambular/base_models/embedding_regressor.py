import lightning as pl
import torch
import torch.nn as nn

from ..utils.config import MambularConfig
from ..utils.mamba_arch import Mamba
from ..utils.mlp_utils import MLP


class BaseEmbeddingMambularRegressor(pl.LightningModule):
    """
    A specialized regression module for protein data, built on PyTorch Lightning and integrating the Mamba architecture.
    It supports embeddings for categorical features and can process raw or embedded numerical features, making it suitable
    for complex protein sequence data.

    Parameters
    ----------
    config : MambularConfig
        Configuration parameters for the model architecture.
    cat_feature_info : dict, optional
        Information about categorical features, mapping feature names to the number of unique categories. Defaults to None.
    num_feature_info : dict, optional
        Information about numerical features, mapping feature names to their number of dimensions after embedding. Defaults to None.
    lr : float, optional
        Learning rate for the optimizer. Defaults to 1e-03.
    lr_patience : int, optional
        Number of epochs with no improvement after which learning rate will be reduced. Defaults to 10.
    weight_decay : float, optional
        Weight decay coefficient for regularization in the optimizer. Defaults to 0.025.
    lr_factor : float, optional
        Factor by which the learning rate will be reduced by the scheduler. Defaults to 0.75.
    seq_size : int, optional
        Size of sequence chunks for processing numerical features. Relevant when `raw_embeddings` is False.
    raw_embeddings : bool, optional
        Indicates whether to use raw numerical features directly or to process them into embeddings. Defaults to False.


    Attributes
    ----------
    mamba : Mamba
        The core neural network module implementing the Mamba architecture.
    norm_f : nn.Module
        Normalization layer applied after the Mamba block.
    tabular_head : nn.Linear
        Final linear layer mapping the features to the regression target.
    loss_fct : nn.MSELoss
        The loss function for regression tasks.
    """

    def __init__(
        self,
        config: MambularConfig,
        cat_feature_info: dict = None,
        num_feature_info: dict = None,
        lr=1e-03,
        lr_patience=10,
        weight_decay=0.025,
        lr_factor=0.75,
        seq_size: int = 20,
        raw_embeddings=False,
        head_layer_sizes=[64, 32, 32],
        head_dropout: float = 0.3,
        head_skip_layers: bool = False,
        head_activation="leakyrelu",
        head_use_batch_norm: bool = False,
        attn_dropout: float = 0.3,
    ):
        super().__init__()

        self.config = config
        self.lr = lr
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.seq_size = seq_size
        self.raw_embeddings = raw_embeddings

        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "softplus": nn.Softplus(),
            "leakyrelu": nn.LeakyReLU(),
            "linear": nn.Identity(),
        }

        self.embedding_activation = activations.get(
            self.config.num_embedding_activation.lower()
        )
        if self.embedding_activation is None:
            raise ValueError(
                f"Unsupported activation function: {self.config.num_embedding_activation}"
            )

        if not self.raw_embeddings:
            data_size = len(num_feature_info.items())
            num_embedding_modules = data_size // self.seq_size
            self.num_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.seq_size, self.config.d_model, bias=False),
                        # Example using ReLU as the activation function, change as needed
                        self.embedding_activation,
                    )
                    for _ in range(num_embedding_modules)
                ]
            )
        else:
            data_size = len(num_feature_info.items())
            num_embedding_modules = data_size
            self.num_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(1, self.config.d_model, bias=False),
                        # Example using ReLU as the activation function, change as needed
                        self.embedding_activation,
                    )
                    for _ in range(num_embedding_modules)
                ]
            )

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories + 1, self.config.d_model)
                for feature_name, num_categories in cat_feature_info.items()
            ]
        )

        self.mamba = Mamba(self.config)
        self.norm_f = self.config.norm(self.config.d_model)
        head_activation = activations.get(head_activation.lower(), nn.Identity())

        # Combine all layers into a Sequential module
        self.tabular_head = MLP(
            self.config.d_model,
            hidden_units_list=head_layer_sizes,
            dropout_rate=head_dropout,
            use_skip_layers=head_skip_layers,
            activation_fn=head_activation,
            use_batch_norm=head_use_batch_norm,
        )

        self.pooling_method = self.config.pooling_method
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))

        if self.config.layer_norm_after_embedding:
            self.embedding_norm = nn.LayerNorm(self.config.d_model)

        self.loss_fct = nn.MSELoss()

    def forward(self, cat_features, num_features):
        """
        Defines the forward pass of the model, processing both categorical and numerical features,
        and returning regression predictions.

        Parameters
        ----------
        cat_features : Tensor
            Tensor containing the categorical features.
        num_features : Tensor
            Tensor containing the numerical features or raw sequence data, depending on `raw_embeddings`.


        Returns
        -------
        Tensor
            The output predictions of the model for regression tasks.
        """
        batch_size = (
            cat_features[0].size(0) if cat_features != [] else num_features[0].size(0)
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Process categorical features if present
        if not self.raw_embeddings:
            if len(self.cat_embeddings) > 0 and cat_features:
                cat_embeddings = [
                    emb(cat_features[i]) for i, emb in enumerate(self.cat_embeddings)
                ]
                cat_embeddings = torch.stack(cat_embeddings, dim=1)
                cat_embeddings = torch.squeeze(cat_embeddings, dim=2)
            else:
                cat_embeddings = None

            if len(self.num_embeddings) > 0 and num_features:
                num_embeddings = []
                # Iterate through the num_embeddings, taking slices of num_features for each
                for i, emb in enumerate(self.num_embeddings):
                    # Calculate start and end indices for slicing the list
                    start_idx = i * self.seq_size
                    end_idx = start_idx + self.seq_size

                    # Slice the num_features list to get the current chunk
                    current_chunk = num_features[start_idx:end_idx]

                    # If the current_chunk is not empty, process it
                    if current_chunk:
                        # Concatenate tensors in the current chunk along dimension 1
                        chunk_tensor = torch.cat(current_chunk, dim=1)
                        # Apply the embedding layer to the chunk_tensor
                        num_embeddings.append(emb(chunk_tensor))

                # Stack the resulting embeddings along the second dimension if num_embeddings is not empty
                if num_embeddings:
                    num_embeddings = torch.stack(num_embeddings, dim=1)
            else:
                num_embeddings = None

        else:
            # Process categorical features if present
            if len(self.cat_embeddings) > 0 and cat_features:
                cat_embeddings = [
                    emb(cat_features[i]) for i, emb in enumerate(self.cat_embeddings)
                ]
                cat_embeddings = torch.stack(cat_embeddings, dim=1)
                cat_embeddings = torch.squeeze(cat_embeddings, dim=2)
                if self.config.layer_norm_after_embedding:
                    cat_embeddings = self.embedding_norm(cat_embeddings)
            else:
                cat_embeddings = None

            # Process numerical features if present
            if len(self.num_embeddings) > 0 and num_features:
                num_embeddings = [
                    emb(num_features[i]) for i, emb in enumerate(self.num_embeddings)
                ]
                num_embeddings = torch.stack(num_embeddings, dim=1)
                if self.config.layer_norm_after_embedding:
                    num_embeddings = self.embedding_norm(num_embeddings)
            else:
                num_embeddings = None

        # Combine embeddings if both types are present, otherwise use whichever is available
        if cat_embeddings is not None and num_embeddings is not None:
            x = torch.cat([cls_tokens, cat_embeddings, num_embeddings], dim=1)
        elif cat_embeddings is not None:
            x = torch.cat([cls_tokens, cat_embeddings], dim=1)
        elif num_embeddings is not None:
            x = torch.cat([cls_tokens, num_embeddings], dim=1)
        else:
            raise ValueError("No features provided to the model.")

        x = self.mamba(x)

        # Apply pooling based on the specified method
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

    def training_step(self, batch, batch_idx):
        """
        Processes a single batch during training, computes the loss, and logs training metrics.

        Parameters
        ----------
        batch : tuple
            A batch of data from the DataLoader, containing numerical features, categorical features, and labels.
        batch_idx : int
            The index of the batch within the epoch.


        Returns
        -------
        Tensor
            The computed loss for the batch.
        """
        num_features, cat_features, labels = batch
        preds = self(num_features, cat_features)

        loss = self.loss_fct(preds.squeeze(), labels.float())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Processes a single batch during validation, computes the loss, and logs validation metrics.

        Parameters
        ----------
        batch : tuple
            A batch of data from the DataLoader, containing numerical features, categorical features, and labels.
        batch_idx : int
            The index of the batch within the epoch.
        """
        num_features, cat_features, labels = batch
        preds = self(num_features, cat_features)

        loss = self.loss_fct(preds.squeeze(), labels.float())
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        """
        Sets up the model's optimizer and learning rate scheduler based on the configurations provided.

        Returns
        -------
        dict
            A dictionary containing the optimizer and lr_scheduler configurations.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.config.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_factor,
                patience=self.lr_patience,
                verbose=True,
            ),
            "monitor": "val_loss",  # Name of the metric to monitor
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
