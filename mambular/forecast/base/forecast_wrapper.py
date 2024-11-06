import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from typing import Type


class ForecastingTaskModel(pl.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating a forecasting model.

    Parameters
    ----------
    model_class : Type[nn.Module]
        The model class to be instantiated and trained.
    config : dataclass
        Configuration dataclass containing model hyperparameters.
    loss_fn : callable, optional
        Loss function to be used during training and evaluation.
    time_steps : int
        Number of past time steps to use as input.
    forecast_horizon : int
        Number of future time steps to predict.
    lr : float, optional
        Learning rate for the optimizer (default is 1e-3).
    optimizer_type : str, optional
        Type of optimizer to use (default is "Adam").
    optimizer_args : dict, optional
        Additional arguments for the optimizer.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        config,
        cat_feature_info,
        num_feature_info,
        early_pruning_threshold=None,
        pruning_epoch=5,
        num_classes=1,
        optimizer_type: str = "Adam",
        optimizer_args: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.optimizer_type = optimizer_type
        self.num_classes = num_classes
        self.early_pruning_threshold = early_pruning_threshold
        self.pruning_epoch = pruning_epoch
        self.val_losses = []

        self.optimizer_params = {
            k.replace("optimizer_", ""): v
            for k, v in optimizer_args.items()
            if k.startswith("optimizer_")
        }

        self.save_hyperparameters(ignore=["model_class", "loss_fn"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        self.base_model = model_class(
            config=config,
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            num_classes=self.num_classes,
            **kwargs,
        )
        # Define metrics for forecasting
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)  # RMSE

        self.loss_fn = nn.MSELoss()

    def forward(self, num_features, cat_features):
        """
        Forward pass through the forecasting model.

        Parameters
        ----------
        x : Tensor
            Input tensor with past time steps.

        Returns
        -------
        Tensor
            Model predictions for future time steps.
        """
        return self.base_model(num_features=num_features, cat_features=cat_features)

    def compute_loss(self, predictions, targets):
        """
        Compute the loss for the given predictions and true labels.

        Parameters
        ----------
        predictions : Tensor
            Model predictions.
        targets : Tensor
            True future values.

        Returns
        -------
        Tensor
            Computed loss.
        """
        return self.loss_fn(predictions, targets)

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing historical and target sequences.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Training loss.
        """
        cat_features, num_features, targets = batch
        if targets.dim() == 3:
            targets = targets.squeeze(-1)
        preds = self(num_features=num_features, cat_features=cat_features)
        loss = self.compute_loss(preds, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing historical and target sequences.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Validation loss.
        """
        cat_features, num_features, targets = batch
        if targets.dim() == 3:
            targets = targets.squeeze(-1)
        preds = self(num_features=num_features, cat_features=cat_features)
        val_loss = self.compute_loss(preds, targets)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        # Log metrics
        self.log("val_mae", self.mae(preds, targets), on_epoch=True, prog_bar=True)
        self.log("val_rmse", self.rmse(preds, targets), on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing historical and target sequences.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Test loss.
        """
        cat_features, num_features, targets = batch
        if targets.dim() == 3:
            targets = targets.squeeze(-1)
        preds = self(num_features=num_features, cat_features=cat_features)
        test_loss = self.compute_loss(preds, targets)
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)

        # Log metrics
        self.log("test_mae", self.mae(preds, targets), on_epoch=True, prog_bar=True)
        self.log("test_rmse", self.rmse(preds, targets), on_epoch=True, prog_bar=True)
        return test_loss

    def predict_step(self, batch, batch_idx):
        """
        Test step for a single batch.

        Parameters
        ----------
        batch : tuple
            Batch of data containing historical and target sequences.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Tensor
            Test loss.
        """
        cat_features, num_features, targets = batch
        if targets.dim() == 3:
            targets = targets.squeeze(-1)
        preds = self(num_features=num_features, cat_features=cat_features)

        return preds

    def configure_optimizers(self):
        """
        Sets up the model's optimizer and learning rate scheduler based on the configurations provided.
        """
        optimizer_class = getattr(torch.optim, self.optimizer_type)
        optimizer = optimizer_class(
            self.base_model.parameters(), lr=self.lr, **self.optimizer_params
        )

        # Scheduler (optional, adjust as needed)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min"
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
