import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from typing import Type


class TaskModel(pl.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating a model.

    Parameters
    ----------
    model_class : Type[nn.Module]
        The model class to be instantiated and trained.
    config : dataclass
        Configuration dataclass containing model hyperparameters.
    loss_fn : callable
        Loss function to be used during training and evaluation.
    lr : float, optional
        Learning rate for the optimizer (default is 1e-3).
    num_classes : int, optional
        Number of classes for classification tasks (default is 1).
    lss : bool, optional
        Custom flag for additional loss configuration (default is False).
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        config,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        lss=False,
        family=None,
        loss_fct: callable = None,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lss = lss
        self.family = family
        self.loss_fct = loss_fct

        if lss:
            pass
        else:
            if num_classes == 2:
                if not self.loss_fct:
                    self.loss_fct = nn.BCEWithLogitsLoss()
                self.acc = torchmetrics.Accuracy(task="binary")
                self.auroc = torchmetrics.AUROC(task="binary")
                self.precision = torchmetrics.Precision(task="binary")
                self.num_classes = 1
            elif num_classes > 2:
                if not self.loss_fct:
                    self.loss_fct = nn.CrossEntropyLoss()
                self.acc = torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
                self.auroc = torchmetrics.AUROC(
                    task="multiclass", num_classes=num_classes
                )
                self.precision = torchmetrics.Precision(
                    task="multiclass", num_classes=num_classes
                )
            else:
                self.loss_fct = nn.MSELoss()

        self.save_hyperparameters(ignore=["model_class", "loss_fn"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        if family is None and num_classes == 2:
            output_dim = 1
        else:
            output_dim = num_classes

        self.model = model_class(
            config=config,
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            num_classes=output_dim,
            **kwargs,
        )

    def forward(self, num_features, cat_features):
        """
        Forward pass through the model.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the model's forward method.
        **kwargs : dict
            Keyword arguments passed to the model's forward method.

        Returns
        -------
        Tensor
            Model output.
        """

        return self.model.forward(num_features, cat_features)

    def compute_loss(self, predictions, y_true):
        """
        Compute the loss for the given predictions and true labels.

        Parameters
        ----------
        predictions : Tensor
            Model predictions.
        y_true : Tensor
            True labels.

        Returns
        -------
        Tensor
            Computed loss.
        """
        if self.lss:
            return self.family.compute_loss(predictions, y_true)
        else:
            loss = self.loss_fct(predictions, y_true)
            return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Training loss.
        """

        cat_features, num_features, labels = batch
        preds = self(num_features=num_features, cat_features=cat_features)
        loss = self.compute_loss(preds, labels)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # Log additional metrics
        if not self.lss:
            if self.num_classes > 1:
                acc = self.acc(preds, labels)
                self.log(
                    "train_acc",
                    acc,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
            elif isinstance(self.loss_fct, nn.MSELoss):
                rmse = torch.sqrt(loss)
                self.log(
                    "train_rmse",
                    rmse,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Validation loss.
        """

        cat_features, num_features, labels = batch
        preds = self(num_features=num_features, cat_features=cat_features)
        val_loss = self.compute_loss(preds, labels)

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Log additional metrics
        if not self.lss:
            if self.num_classes > 1:
                acc = self.acc(preds, labels)
                self.log(
                    "val_acc",
                    acc,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
            elif isinstance(self.loss_fct, nn.MSELoss):
                rmse = torch.sqrt(val_loss)
                self.log(
                    "val_rmse",
                    rmse,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing numerical features, categorical features, and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Test loss.
        """
        cat_features, num_features, labels = batch
        preds = self(num_features=num_features, cat_features=cat_features)
        test_loss = self.compute_loss(preds, labels)

        self.log(
            "test_loss",
            test_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Log additional metrics
        if not self.lss:
            if self.num_classes > 1:
                acc = self.acc(preds, labels)
                self.log(
                    "test_acc",
                    acc,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
            elif isinstance(self.loss_fct, nn.MSELoss):
                rmse = torch.sqrt(test_loss)
                self.log(
                    "test_rmse",
                    rmse,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return test_loss

    def configure_optimizers(self):
        """
        Sets up the model's optimizer and learning rate scheduler based on the configurations provided.

        Returns
        -------
        dict
            A dictionary containing the optimizer and lr_scheduler configurations.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_factor,
                patience=self.lr_patience,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
