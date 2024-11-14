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
        early_pruning_threshold=None,
        pruning_epoch=5,
        optimizer_type: str = "Adam",
        optimizer_args: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.optimizer_type = optimizer_type
        self.num_classes = num_classes
        self.lss = lss
        self.family = family
        self.loss_fct = loss_fct
        self.early_pruning_threshold = early_pruning_threshold
        self.pruning_epoch = pruning_epoch
        self.val_losses = []

        self.optimizer_params = {
            k.replace("optimizer_", ""): v
            for k, v in optimizer_args.items()
            if k.startswith("optimizer_")
        }

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

        self.base_model = model_class(
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

        return self.base_model.forward(num_features, cat_features)

    def compute_loss(self, predictions, y_true):
        """
        Compute the loss for the given predictions and true labels.

        Parameters
        ----------
        predictions : Tensor
            Model predictions. Shape: (batch_size, k, output_dim) for ensembles, or (batch_size, output_dim) otherwise.
        y_true : Tensor
            True labels. Shape: (batch_size, output_dim).

        Returns
        -------
        Tensor
            Computed loss.
        """
        if self.lss:
            if getattr(self.base_model, "returns_ensemble", False):
                loss = 0.0
                for ensemble_member in range(predictions.shape[1]):
                    loss += self.family.compute_loss(
                        predictions[:, ensemble_member], y_true.squeeze(-1)
                    )
                return loss
            else:
                return self.family.compute_loss(predictions, y_true.squeeze(-1))

        if getattr(self.base_model, "returns_ensemble", False):  # Ensemble case
            if (
                self.loss_fct.__class__.__name__ == "CrossEntropyLoss"
                and predictions.dim() == 3
            ):
                # Classification case with ensemble: predictions (N, E, k), y_true (N,)
                N, E, k = predictions.shape
                loss = 0.0
                for ensemble_member in range(E):
                    loss += self.loss_fct(predictions[:, ensemble_member, :], y_true)
                return loss

            else:
                # Regression case with ensemble (e.g., MSE) or other compatible losses
                y_true_expanded = y_true.expand_as(predictions)
                return self.loss_fct(predictions, y_true_expanded)
        else:
            # Non-ensemble case
            return self.loss_fct(predictions, y_true)

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch, incorporating penalty if the model has a penalty_forward method.

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

        # Check if the model has a `penalty_forward` method
        if hasattr(self.base_model, "penalty_forward"):
            preds, penalty = self.base_model.penalty_forward(
                num_features=num_features, cat_features=cat_features
            )
            loss = self.compute_loss(preds, labels) + penalty
        else:
            preds = self(num_features=num_features, cat_features=cat_features)
            loss = self.compute_loss(preds, labels)

        # Log the training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # Log additional metrics
        if not self.lss and not self.base_model.returns_ensemble:
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
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Log additional metrics
        if not self.lss and not self.base_model.returns_ensemble:
            if self.num_classes > 1:
                acc = self.acc(preds, labels)
                self.log(
                    "val_acc",
                    acc,
                    on_step=False,
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
        if not self.lss and not self.base_model.returns_ensemble:
            if self.num_classes > 1:
                acc = self.acc(preds, labels)
                self.log(
                    "test_acc",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return test_loss

    def on_validation_epoch_end(self):
        """
        Callback executed at the end of each validation epoch.

        This method retrieves the current validation loss from the trainer's callback metrics
        and stores it in a list for tracking validation losses across epochs. It also applies
        pruning logic to stop training early if the validation loss exceeds a specified threshold.

        Parameters
        ----------
        None

        Attributes
        ----------
        val_loss : torch.Tensor or None
            The validation loss for the current epoch, retrieved from `self.trainer.callback_metrics`.
        val_loss_value : float
            The validation loss for the current epoch, converted to a float.
        val_losses : list of float
            A list storing the validation losses for each epoch.
        pruning_epoch : int
            The epoch after which pruning logic will be applied.
        early_pruning_threshold : float, optional
            The threshold for early pruning based on validation loss. If the current validation
            loss exceeds this value, training will be stopped early.

        Notes
        -----
        If the current epoch is greater than or equal to `pruning_epoch`, and the validation
        loss exceeds the `early_pruning_threshold`, the training is stopped early by setting
        `self.trainer.should_stop` to True.
        """
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss_value = val_loss.item()
            self.val_losses.append(val_loss_value)  # Store val_loss for each epoch

            # Apply pruning logic if needed
            if self.current_epoch >= self.pruning_epoch:
                if (
                    self.early_pruning_threshold is not None
                    and val_loss_value > self.early_pruning_threshold
                ):
                    print(
                        f"Pruned at epoch {self.current_epoch}, val_loss {val_loss_value}"
                    )
                    self.trainer.should_stop = True  # Stop training early

    def epoch_val_loss_at(self, epoch):
        """
        Retrieve the validation loss at a specific epoch.

        This method allows the user to query the validation loss for any given epoch,
        provided the epoch exists within the range of completed epochs. If the epoch
        exceeds the length of the `val_losses` list, a default value of infinity is returned.

        Parameters
        ----------
        epoch : int
            The epoch number for which the validation loss is requested.

        Returns
        -------
        float
            The validation loss for the requested epoch. If the epoch does not exist,
            the method returns `float("inf")`.

        Notes
        -----
        This method relies on `self.val_losses` which stores the validation loss values
        at the end of each epoch during training.
        """
        if epoch < len(self.val_losses):
            return self.val_losses[epoch]
        else:
            return float("inf")

    def configure_optimizers(self):
        """
        Sets up the model's optimizer and learning rate scheduler based on the configurations provided.
        The optimizer type can be chosen by the user (Adam, SGD, etc.).
        """
        # Dynamically choose the optimizer based on the passed optimizer_type
        optimizer_class = getattr(torch.optim, self.optimizer_type)

        # Initialize the optimizer with the chosen class and parameters
        optimizer = optimizer_class(
            self.base_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.optimizer_params,  # Pass any additional optimizer-specific parameters
        )

        # Define learning rate scheduler
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
