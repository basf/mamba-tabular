import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from ..utils.config import MambularConfig
from ..utils.mamba_arch import Mamba


class BaseMambularClassifier(pl.LightningModule):
    """
    A base class for building classification models using the Mambular architecture within the PyTorch Lightning framework.

    This class integrates various components such as embeddings for categorical and numerical features, the Mambular model
    for processing sequences of embeddings, and a classification head for prediction. It supports multi-class and binary classification tasks.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task. For binary classification, this should be 2.
    config : MambularConfig
        An instance of MambularConfig containing configuration parameters for the Mambular model.
    cat_feature_info : dict, optional
        A dictionary mapping the names of categorical features to their number of unique categories.
        This information is used to configure embedding layers for categorical features. Defaults to None.
    num_feature_info : dict, optional
        A dictionary mapping the names of numerical features to the size of their input dimensions.
        This information is used to configure embedding layers for numerical features. Defaults to None.
    lr : float, optional
        The learning rate for the optimizer. Defaults to 1e-03.
    lr_patience : int, optional
        The number of epochs with no improvement after which learning rate will be reduced. Defaults to 10.
    weight_decay : float, optional
        Weight decay (L2 penalty) parameter for the optimizer. Defaults to 0.025.
    lr_factor : float, optional
        Factor by which the learning rate will be reduced. Defaults to 0.75.


    Attributes
    ----------
    embedding_activation : nn.Module
        The activation function to be applied after the linear transformation of numerical features.
    num_embeddings : nn.ModuleList
        A list of sequential modules, each corresponding to an embedding layer for a numerical feature.
    cat_embeddings : nn.ModuleList
        A list of embedding layers, each corresponding to a categorical feature.
    mamba : Mamba
        The Mambular model for processing sequences of embeddings.
    norm_f : nn.Module
        A normalization layer applied after the Mambular model.
    tabular_head : nn.Linear
        A linear layer for predicting the class labels from the aggregated embedding representation.
    pooling_method : str
        The method used to aggregate embeddings across features. Supported methods are 'avg', 'max', and 'sum'.
    loss_fct : nn.Module
        The loss function used for training the model, configured based on the number of classes.
    acc : torchmetrics.Accuracy
        A metric for computing the accuracy of predictions.
    auroc : torchmetrics.AUROC
        A metric for computing the Area Under the Receiver Operating Characteristic curve.
    precision : torchmetrics.Precision
        A metric for computing the precision of predictions.

    """

    def __init__(
        self,
        num_classes,
        config: MambularConfig,
        cat_feature_info: dict = None,
        num_feature_info: dict = None,
        lr=1e-03,
        lr_patience=10,
        weight_decay=0.025,
        lr_factor=0.75,
    ):
        super().__init__()

        self.config = config
        self.num_classes = 1 if num_classes == 2 else num_classes
        self.lr = lr
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

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

        self.num_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_shape, self.config.d_model, bias=False),
                    nn.BatchNorm1d(self.config.d_model),
                    # Example using ReLU as the activation function, change as needed
                    self.embedding_activation,
                )
                for feature_name, input_shape in num_feature_info.items()
            ]
        )

        # Create embedding layers for categorical features based on cat_feature_info
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories + 1, self.config.d_model)
                for feature_name, num_categories in cat_feature_info.items()
            ]
        )

        self.mamba = Mamba(self.config)
        self.norm_f = self.config.norm(self.config.d_model)

        mlp_activation_fn = activations.get(
            self.config.tabular_head_activation.lower(), nn.Identity()
        )
        mlp_layers = []
        input_dim = self.config.d_model  # Initial input dimension

        # Iterate over the specified units for each layer in the MLP
        for units in self.config.tabular_head_units:
            mlp_layers.append(nn.Linear(input_dim, units))
            mlp_layers.append(mlp_activation_fn)
            mlp_layers.append(nn.Dropout(self.config.tabular_head_dropout))
            input_dim = units

        # Add the final linear layer to map to a single output value
        mlp_layers.append(nn.Linear(input_dim, self.num_classes))

        # Combine all layers into a Sequential module
        self.tabular_head = nn.Sequential(*mlp_layers)

        self.pooling_method = self.config.pooling_method
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))

        if self.num_classes > 2:
            self.loss_fct = nn.CrossEntropyLoss()
            self.acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes
            )
            self.auroc = torchmetrics.AUROC(
                task="multiclass", num_classes=self.num_classes
            )
            self.precision = torchmetrics.Precision(
                task="multiclass", num_classes=self.num_classes
            )
        else:
            self.loss_fct = torch.nn.BCEWithLogitsLoss()
            self.acc = torchmetrics.Accuracy(task="binary")
            self.auroc = torchmetrics.AUROC(task="binary")
            self.precision = torchmetrics.Precision(task="binary")

    def forward(self, cat_features, num_features):
        """
        Defines the forward pass of the classifier.

        Parameters
        ----------
        cat_features : Tensor
            Tensor containing the categorical features.
        num_features : Tensor
            Tensor containing the numerical features.


        Returns
        -------
        Tensor
            The output predictions of the model.
        """
        batch_size = (
            cat_features[0].size(0)
            if cat_features is not None
            else num_features[0].size(0)
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Process categorical features if present
        if len(self.cat_embeddings) > 0 and cat_features:
            cat_embeddings = [
                emb(cat_features[i]) for i, emb in enumerate(self.cat_embeddings)
            ]
            cat_embeddings = torch.stack(cat_embeddings, dim=1)
            cat_embeddings = torch.squeeze(cat_embeddings, dim=2)
        else:
            cat_embeddings = None

        # Process numerical features if present
        if len(self.num_embeddings) > 0 and num_features:
            num_embeddings = [
                emb(num_features[i]) for i, emb in enumerate(self.num_embeddings)
            ]
            num_embeddings = torch.stack(num_embeddings, dim=1)
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
        elif self.pooling_method == "cls":
            x = x[:, 0]
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")

        x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds

    def training_step(self, batch, batch_idx):
        """
        Processes a single batch during training, computes the loss and logs training metrics.

        Parameters
        ----------
        batch : tuple
            A batch of data from the DataLoader, containing numerical features, categorical features, and labels.
        batch_idx : int
            The index of the batch within the epoch.
        """

        num_features, cat_features, labels = batch
        preds = self(num_features, cat_features)

        if self.num_classes == 1:
            labels = labels.unsqueeze(
                1
            ).float()  # Reshape for binary classification loss calculation

        loss = self.loss_fct(preds, labels)
        self.log("train_loss", loss)
        # Calculate and log training accuracy

        acc = self.acc(preds, labels.int())
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Calculate and log AUROC
        auroc = self.auroc(preds, labels.int())
        self.log(
            "train_auroc",
            auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Calculate and log precision
        precision = self.precision(preds, labels.int())
        self.log(
            "train_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Processes a single batch during validation, computes the loss and logs validation metrics.

        Parameters
        ----------
        batch : tuple
            A batch of data from the DataLoader, containing numerical features, categorical features, and labels.
        batch_idx : int
            The index of the batch within the epoch.
        """

        num_features, cat_features, labels = batch
        preds = self(num_features, cat_features)

        if self.num_classes == 1:
            labels = labels.unsqueeze(
                1
            ).float()  # Reshape for binary classification loss calculation

        loss = self.loss_fct(preds, labels)
        self.log("val_loss", loss)
        # Calculate and log training accuracy

        acc = self.acc(preds, labels.int())
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        auroc = self.auroc(preds, labels.int())
        self.log(
            "val_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Calculate and log precision
        precision = self.precision(preds, labels.int())
        self.log(
            "val_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

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
