import torch
import pandas as pd
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
from .forecast_dataset import ForecastMambularDataset


class ForecastMambularDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning data module for time series forecasting with options for temporal, lagged, and rolling features,
    and flexible forecast horizon configurations.

    Parameters:
        preprocessor: object
            An instance of your preprocessor class.
        batch_size: int
            Batch size for the DataLoader.
        time_steps: int
            Number of historical time steps to use as input.
        forecast_horizon: int
            Number of future time steps to predict.
        data: DataFrame
            Full time series dataset with both features and target(s).
        val_size: float, optional
            Proportion of data to use in validation if no validation set is provided.
        random_state: int, optional
            Random seed for reproducibility in data splitting.
        include_timestamps: bool, optional
            Whether to include timestamps for time-based features.
        include_time_embeddings: bool, optional
            Whether to include cyclical time embeddings (e.g., day of week).
        include_lagged_features: list of int, optional
            Lags to include as additional numerical features.
        include_rolling_stats: dict, optional
            Specifies rolling window sizes and statistics (e.g., {'window_size': [mean, std]}).
        external_features: dict, optional
            External time-dependent features (e.g., weather).
        scaling: str, optional
            Type of scaling to apply ("standard", "minmax" or None).
        variable_horizon: bool, optional
            If True, dynamically adjusts forecast horizon.
    """

    def __init__(
        self,
        preprocessor,
        batch_size,
        time_steps,
        forecast_horizon,
        data,
        val_size=0.2,
        random_state=101,
        include_timestamps=False,
        include_time_embeddings=False,
        include_lagged_features=None,
        include_rolling_stats=None,
        include_holiday_indicator=False,
        external_features=None,
        scaling=None,
        variable_horizon=False,
        **dataloader_kwargs,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.forecast_horizon = forecast_horizon
        self.data = data
        self.val_size = val_size
        self.random_state = random_state
        self.include_timestamps = include_timestamps
        self.include_time_embeddings = include_time_embeddings
        self.include_lagged_features = include_lagged_features
        self.include_rolling_stats = include_rolling_stats
        self.include_holiday_indicator = include_holiday_indicator
        self.external_features = external_features
        self.scaling = scaling
        self.variable_horizon = variable_horizon
        self.dataloader_kwargs = dataloader_kwargs

    def preprocess_data(self):
        """
        Preprocesses the dataset with a sequential train-validation split.
        """
        split_idx = int(len(self.data) * (1 - self.val_size))
        train_data, val_data = self.data.iloc[:split_idx], self.data.iloc[split_idx:]

        # Fit the preprocessor on training data only
        self.preprocessor.fit(train_data)
        (
            self.cat_feature_info,
            self.num_feature_info,
        ) = self.preprocessor.get_feature_info()

        # Transform the data
        train_preprocessed = self.preprocessor.transform(train_data)
        val_preprocessed = self.preprocessor.transform(val_data)

        # Convert preprocessed data to tensors
        self.train_cat_tensors, self.train_num_tensors = self._convert_to_tensor(
            train_preprocessed
        )
        self.val_cat_tensors, self.val_num_tensors = self._convert_to_tensor(
            val_preprocessed
        )

        # Extract timestamps if enabled
        self.train_timestamps = train_data.index if self.include_timestamps else None
        self.val_timestamps = val_data.index if self.include_timestamps else None

    def preprocess_test_data(self, test_data):
        test_preprocessed = self.preprocessor.transform(test_data)
        self.test_cat_tensors, self.test_num_tensors = self._convert_to_tensor(
            test_preprocessed
        )
        self.test_timestamps = test_data.index if self.include_timestamps else None

    def setup(self, stage: str):
        """
        Sets up datasets for training, validation, and testing.
        """
        if stage == "fit":
            self.preprocess_data()

            self.train_dataset = ForecastMambularDataset(
                cat_features_list=self.train_cat_tensors,
                num_features_list=self.train_num_tensors,
                time_steps=self.time_steps,
                forecast_horizon=self.forecast_horizon,
                timestamps=self.train_timestamps,
                include_time_embeddings=self.include_time_embeddings,
                include_lagged_features=self.include_lagged_features,
                include_rolling_stats=self.include_rolling_stats,
                include_holiday_indicator=self.include_holiday_indicator,
                external_features=self.external_features,
                scaling=self.scaling,
            )

            self.val_dataset = ForecastMambularDataset(
                cat_features_list=self.val_cat_tensors,
                num_features_list=self.val_num_tensors,
                time_steps=self.time_steps,
                forecast_horizon=self.forecast_horizon,
                timestamps=self.val_timestamps,
                include_time_embeddings=self.include_time_embeddings,
                include_lagged_features=self.include_lagged_features,
                include_rolling_stats=self.include_rolling_stats,
                include_holiday_indicator=self.include_holiday_indicator,
                external_features=self.external_features,
                scaling=self.scaling,
            )

        elif stage == "test":
            self.test_dataset = ForecastMambularDataset(
                cat_features_list=self.test_cat_tensors,
                num_features_list=self.test_num_tensors,
                time_steps=self.time_steps,
                forecast_horizon=self.forecast_horizon,
                timestamps=self.test_timestamps,
                include_time_embeddings=self.include_time_embeddings,
                include_lagged_features=self.include_lagged_features,
                include_rolling_stats=self.include_rolling_stats,
                include_holiday_indicator=self.include_holiday_indicator,
                external_features=self.external_features,
                scaling=self.scaling,
            )

    def _convert_to_tensor(self, data):
        """
        Converts preprocessed data into lists of categorical and numerical feature tensors.
        """
        cat_tensors, num_tensors = [], []

        for key in self.cat_feature_info:
            cat_key = "cat_" + key
            if cat_key in data:
                cat_tensors.append(torch.tensor(data[cat_key], dtype=torch.long))

        for key in self.num_feature_info:
            num_key = "num_" + key
            if num_key in data:
                num_tensors.append(torch.tensor(data[num_key], dtype=torch.float32))

        return cat_tensors, num_tensors

    def train_dataloader(self):
        """
        Returns the training DataLoader with sequential ordering.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        """
        Returns the validation DataLoader with sequential ordering.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        """
        Returns the test DataLoader with sequential ordering.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_dataset.time_steps,
            shuffle=False,
            **self.dataloader_kwargs,
        )

    def get_preprocessor(self):
        """Returns the preprocessor with scaling parameters for rescaling in evaluation."""
        return self.preprocessor
