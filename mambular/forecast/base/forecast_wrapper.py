import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from typing import Type, Optional, Dict, Callable
import numpy as np
import pandas as pd


class ForecastingTaskModel(pl.LightningModule):
    """
    PyTorch Lightning Module for a flexible time series forecasting model with added support
    for temporal embeddings, uncertainty estimation, variable self.horizons, custom metrics, and temporal regularization.
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        config,
        cat_feature_info,
        num_feature_info,
        loss_fn: Optional[Callable] = None,
        early_pruning_threshold: Optional[float] = None,
        pruning_epoch: int = 5,
        num_classes: int = 1,
        optimizer_type: str = "Adam",
        optimizer_args: Optional[Dict] = None,
        include_time_embeddings: bool = False,
        include_uncertainty: bool = False,
        forecast_horizon: int = 1,
        lr: float = 1e-3,
        horizons=[1, 2, 3, 5, 10],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_class", "loss_fn", "config"])

        # Set up model parameters
        self.optimizer_type = optimizer_type
        self.num_classes = num_classes
        self.early_pruning_threshold = early_pruning_threshold
        self.pruning_epoch = pruning_epoch
        self.include_time_embeddings = include_time_embeddings
        self.include_uncertainty = include_uncertainty
        self.forecast_horizon = forecast_horizon
        self.horizons = horizons
        self.val_losses = []
        self.lr = lr

        # Model setup
        self.base_model = model_class(
            config=config,
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            num_classes=self.num_classes,
            **kwargs,
        )

        # Loss and metrics
        self.loss_fn = loss_fn or nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)  # RMSE
        self.mase = torchmetrics.MeanAbsoluteError()  # Proxy for MASE calculation

        # Default optimizer parameters
        self.optimizer_params = optimizer_args or {}
        self.lr_scheduler_params = {
            "patience": config.lr_patience,
            "factor": config.lr_factor,
            "threshold": 0.0001,
            "cooldown": 1,
            "min_lr": 1e-6,
        }

    def set_preprocessor(self, preprocessor):
        """Sets the preprocessor to access scaling parameters for rescaling."""
        self.preprocessor = preprocessor

    def forward(self, num_features, cat_features, time_embedding=None):
        """
        Forward pass through the forecasting model.
        """
        inputs = {"num_features": num_features, "cat_features": cat_features}
        if self.include_time_embeddings and time_embedding is not None:
            inputs["time_embedding"] = time_embedding
        return self.base_model(**inputs)

    def compute_loss(self, predictions, targets):
        """
        Compute the loss for the given predictions and true labels.
        """
        if self.include_uncertainty:
            preds_mean, preds_var = predictions
            mse_loss = self.loss_fn(preds_mean, targets)
            variance_loss = torch.mean(preds_var)
            return (
                mse_loss + 0.1 * variance_loss
            )  # 0.1 as weighting for uncertainty penalty
        return self.loss_fn(predictions, targets)

    def compute_mase(self, predictions, targets):
        """
        Compute MASE: Mean Absolute Scaled Error for time series evaluation.
        """
        naive_forecast = targets[..., :-1]  # Shifted version of target series
        scaled_error = torch.mean(torch.abs(targets - predictions)) / torch.mean(
            torch.abs(targets[..., 1:] - naive_forecast) + 1e-08
        )
        return scaled_error

    def step(self, batch, batch_idx):
        cat_features, num_features, targets = batch
        time_embedding = (
            self._create_time_embeddings(num_features)
            if self.include_time_embeddings
            else None
        )
        preds = self(
            num_features=num_features,
            cat_features=cat_features,
            time_embedding=time_embedding,
        )
        loss = self.compute_loss(preds, targets)
        metrics = {
            "mae": self.mae(preds, targets),
            "rmse": self.rmse(preds, targets),
            "mase": self.compute_mase(preds, targets),
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", metrics["mae"], on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", metrics["mae"], on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for evaluating the time series model with iterative forecasting, where each predicted step
        is used as input for the next forecasted step.

        Parameters
        ----------
        batch : tuple
            Batch of data containing historical and target sequences.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary of evaluation metrics for the batch.
        """
        cat_features, num_features, targets = batch
        time_embedding = (
            self._create_time_embeddings(num_features)
            if self.include_time_embeddings
            else None
        )

        # Initialize metrics and storage for recursive predictions
        results = {}
        predictions = []  # To store iterative predictions

        # Initial step: predict t+1 using the true input
        preds = self(
            num_features=num_features,
            cat_features=cat_features,
            time_embedding=time_embedding,
        )
        predictions.append(preds)

        # Recursive prediction for subsequent steps
        current_num_features = [
            nf.clone() for nf in num_features
        ]  # Clone each tensor in num_features list
        for step in range(1, max(self.horizons)):
            # Update each feature in current_num_features with the latest prediction at the last time step
            current_num_features[-1][:, -1] = preds
            preds = self(
                num_features=current_num_features,
                cat_features=cat_features,
                time_embedding=time_embedding,
            )
            predictions.append(preds)

        # Concatenate predictions along the time dimension
        predictions = torch.cat(
            predictions, dim=1
        )  # Shape: (batch_size, max(self.horizons))

        # Evaluate each horizon individually, comparing each step prediction with single-step target
        cat_feature_info, num_feature_info = self.preprocessor.get_feature_info(
            verbose=False
        )
        feature_names = list(
            num_feature_info.keys()
        )  # Extract feature names from num_feature_info

        for horizon in self.horizons:
            if horizon > predictions.size(
                1
            ):  # Skip if horizon exceeds generated predictions
                continue

            # Select predictions for this horizon
            preds_horizon = predictions[
                :,
                horizon - 1,
            ]

            # Construct a DataFrame with one column per feature
            preds_df = pd.DataFrame(preds_horizon.cpu().numpy(), columns=feature_names)

            # Apply inverse transformation
            preds_rescaled_df = self.preprocessor.inverse_transform(preds_df)

            # Convert the rescaled DataFrame back to a tensor for metric calculations
            preds_rescaled = torch.tensor(
                preds_rescaled_df.values, device=preds_horizon.device
            )
            targets_horizon = targets  # Flatten targets to (batch_size,)

            # Basic metrics: MAE, RMSE
            results[
                f"test_mae_{horizon}"
            ] = torchmetrics.functional.mean_absolute_error(
                preds_rescaled, targets_horizon
            )
            results[
                f"test_rmse_{horizon}"
            ] = torchmetrics.functional.mean_squared_error(
                preds_rescaled, targets_horizon, squared=False
            )

            # SMAPE (Symmetric Mean Absolute Percentage Error)
            smape = torch.mean(
                2
                * torch.abs(preds_rescaled - targets_horizon)
                / (torch.abs(preds_rescaled) + torch.abs(targets_horizon) + 1e-8)
            )
            results[f"test_smape_{horizon}"] = smape

            # Coverage for interval-based forecasts (if applicable for probabilistic models)
            if self.include_uncertainty:
                preds_mean, preds_std = (
                    preds_rescaled,
                    preds_rescaled[:, horizon:],  # Assuming std if available
                )
                lower_bound, upper_bound = (
                    preds_mean - 1.96 * preds_std,
                    preds_mean + 1.96 * preds_std,
                )
                coverage = torch.mean(
                    (targets_horizon >= lower_bound)
                    & (targets_horizon <= upper_bound).float()
                )
                results[f"test_coverage_{horizon}"] = coverage

        # Log each metric for the current batch
        for metric_name, value in results.items():
            self.log(metric_name, value, on_epoch=True, prog_bar=True)

        return results

    def predict_step(self, batch, batch_idx):
        cat_features, num_features, _ = batch
        time_embedding = (
            self._create_time_embeddings(num_features)
            if self.include_time_embeddings
            else None
        )
        preds = self(
            num_features=num_features,
            cat_features=cat_features,
            time_embedding=time_embedding,
        )

        cat_feature_info, num_feature_info = self.preprocessor.get_feature_info(
            verbose=False
        )
        feature_names = list(
            num_feature_info.keys()
        )  # Extract feature names from num_feature_info

        # Construct a DataFrame with one column per feature
        preds_df = pd.DataFrame(preds.cpu().numpy(), columns=feature_names)

        # Apply inverse transformation
        preds_rescaled_df = self.preprocessor.inverse_transform(preds_df)

        # Convert the rescaled DataFrame back to a tensor for metric calculations
        preds_rescaled = torch.tensor(preds_rescaled_df.values, device=preds.device)
        return preds_rescaled

    def configure_optimizers(self):
        """
        Set up the optimizer and learning rate scheduler.
        """
        optimizer_class = getattr(torch.optim, self.optimizer_type)
        optimizer = optimizer_class(
            self.base_model.parameters(), lr=self.lr, **self.optimizer_params
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **self.lr_scheduler_params
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _create_time_embeddings(self, num_features):
        """
        Create cyclical embeddings for time-related features (day of the week, month of the year).
        """
        # Example cyclical embeddings for temporal data
        days = torch.arange(0, num_features[0].shape[1]) % 7  # Weekly cycle
        months = torch.arange(0, num_features[0].shape[1]) % 12  # Monthly cycle
        day_embedding = torch.sin(2 * np.pi * days / 7).unsqueeze(0)
        month_embedding = torch.sin(2 * np.pi * months / 12).unsqueeze(0)
        return torch.cat([day_embedding, month_embedding], dim=-1).to(
            num_features.device
        )
