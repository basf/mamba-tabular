import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from ..utils.mamba_arch import Mamba
from ..utils.mlp_utils import MLP
from ..utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
from ..utils.configs import DefaultMambularConfig


class BaseMambularClassifier(pl.LightningModule):
    """
    A base class for building classification models using the Mambular architecture within the PyTorch Lightning framework.

    This class integrates various components such as embeddings for categorical and numerical features, the Mambular model
    for processing sequences of embeddings, and a classification head for prediction. It supports multi-class and binary classification tasks.

    Parameters
    ----------
    num_classes : int
        number of classes for classification.
    cat_feature_info : dict
        Dictionary containing information about categorical features.
    num_feature_info : dict
        Dictionary containing information about numerical features.
    config : DefaultMambularConfig, optional
        Configuration object containing default hyperparameters for the model (default is DefaultMambularConfig()).
    **kwargs : dict
        Additional keyword arguments.


    Attributes
    ----------
    lr : float
        Learning rate.
    lr_patience : int
        Patience for learning rate scheduler.
    weight_decay : float
        Weight decay for optimizer.
    lr_factor : float
        Factor by which the learning rate will be reduced.
    pooling_method : str
        Method to pool the features.
    cat_feature_info : dict
        Dictionary containing information about categorical features.
    num_feature_info : dict
        Dictionary containing information about numerical features.
    embedding_activation : callable
        Activation function for embeddings.
    mamba : Mamba
        Mamba architecture component.
    norm_f : nn.Module
        Normalization layer.
    num_embeddings : nn.ModuleList
        Module list for numerical feature embeddings.
    cat_embeddings : nn.ModuleList
        Module list for categorical feature embeddings.
    tabular_head : MLP
        Multi-layer perceptron head for tabular data.
    cls_token : nn.Parameter
        Class token parameter.
    embedding_norm : nn.Module, optional
        Layer normalization applied after embedding if specified.
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
        cat_feature_info,
        num_feature_info,
        config: DefaultMambularConfig = DefaultMambularConfig(),
        **kwargs,
    ):
        super().__init__()

        self.num_classes = 1 if num_classes == 2 else num_classes
        # Save all hyperparameters
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        # Assigning values from hyperparameters
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

        # Additional layers and components initialization based on hyperparameters
        self.mamba = Mamba(
            d_model=self.hparams.get("d_model", config.d_model),
            n_layers=self.hparams.get("n_layers", config.n_layers),
            expand_factor=self.hparams.get("expand_factor", config.expand_factor),
            bias=self.hparams.get("bias", config.bias),
            d_conv=self.hparams.get("d_conv", config.d_conv),
            conv_bias=self.hparams.get("conv_bias", config.conv_bias),
            dropout=self.hparams.get("dropout", config.dropout),
            dt_rank=self.hparams.get("dt_rank", config.dt_rank),
            d_state=self.hparams.get("d_state", config.d_state),
            dt_scale=self.hparams.get("dt_scale", config.dt_scale),
            dt_init=self.hparams.get("dt_init", config.dt_init),
            dt_max=self.hparams.get("dt_max", config.dt_max),
            dt_min=self.hparams.get("dt_min", config.dt_min),
            dt_init_floor=self.hparams.get("dt_init_floor", config.dt_init_floor),
            norm=globals()[self.hparams.get("norm", config.norm)],
            activation=self.hparams.get("activation", config.activation),
        )

        # Set the normalization layer dynamically
        norm_layer = self.hparams.get("norm", config.norm)
        if norm_layer == "RMSNorm":
            self.norm_f = RMSNorm(self.hparams.get("d_model", config.d_model))
        elif norm_layer == "LayerNorm":
            self.norm_f = LayerNorm(self.hparams.get("d_model", config.d_model))
        elif norm_layer == "BatchNorm":
            self.norm_f = BatchNorm(self.hparams.get("d_model", config.d_model))
        elif norm_layer == "InstanceNorm":
            self.norm_f = InstanceNorm(self.hparams.get("d_model", config.d_model))
        elif norm_layer == "GroupNorm":
            self.norm_f = GroupNorm(1, self.hparams.get("d_model", config.d_model))
        elif norm_layer == "LearnableLayerScaling":
            self.norm_f = LearnableLayerScaling(
                self.hparams.get("d_model", config.d_model)
            )
        else:
            raise ValueError(f"Unsupported normalization layer: {norm_layer}")

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
            self.hparams.get("d_model", config.d_model),
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
            n_output_units=self.num_classes,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.hparams.get("d_model", config.d_model))
        )

        self.loss_fct = nn.MSELoss()

        if self.hparams.get("layer_norm_after_embedding"):
            self.embedding_norm = nn.LayerNorm(
                self.hparams.get("d_model", config.d_model)
            )

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

    def forward(self, num_features, cat_features):
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
            cat_features[0].size(0) if cat_features != [] else num_features[0].size(0)
        )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Process categorical features if present
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

        # Process numerical features if present
        if len(self.num_embeddings) > 0 and num_features:
            num_embeddings = [
                emb(num_features[i]) for i, emb in enumerate(self.num_embeddings)
            ]
            num_embeddings = torch.stack(num_embeddings, dim=1)
            if self.hparams.get("layer_norm_after_embedding"):
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

        cat_features, num_features, labels = batch
        preds = self(num_features=num_features, cat_features=cat_features)

        if self.num_classes == 1:
            labels = torch.float(
                labels.unsqueeze(1), dtype=torch.float32
            )  # Reshape for binary classification loss calculation

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
        cat_features, num_features, labels = batch
        preds = self(num_features=num_features, cat_features=cat_features)

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
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
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
