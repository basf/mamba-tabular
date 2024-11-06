import torch
import pandas as pd
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
from .forecast_dataset import ForecastMambularDataset


class ForecastMambularDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning data module for managing time series data in a structured way.
    Applies sliding window processing for time series forecasting.

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
        **dataloader_kwargs,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.forecast_horizon = forecast_horizon
        self.data = data  # Continuous time series dataset with features and target(s)
        self.val_size = val_size
        self.random_state = random_state
        self.dataloader_kwargs = dataloader_kwargs

    def preprocess_data(self):
        """
        Preprocesses the continuous dataset using a sequential train-validation split.
        """
        # Perform sequential split for time series
        split_idx = int(len(self.data) * (1 - self.val_size))
        train_data, val_data = self.data.iloc[:split_idx], self.data.iloc[split_idx:]

        # Fit the preprocessor on the training data only
        self.preprocessor.fit(train_data)
        self.cat_feature_info, self.num_feature_info = (
            self.preprocessor.get_feature_info()
        )

        # Transform the data
        train_preprocessed = self.preprocessor.transform(train_data)
        val_preprocessed = self.preprocessor.transform(val_data)

        # Convert preprocessed data to tensors for categorical and numerical features
        self.train_cat_tensors, self.train_num_tensors = self._convert_to_tensor(
            train_preprocessed
        )
        self.val_cat_tensors, self.val_num_tensors = self._convert_to_tensor(
            val_preprocessed
        )

    def setup(self, stage: str):
        """
        Create time series datasets for training and validation.
        """
        if stage == "fit":
            self.preprocess_data()

            # Create time series datasets with the transformed tensors
            self.train_dataset = ForecastMambularDataset(
                cat_features_list=self.train_cat_tensors,
                num_features_list=self.train_num_tensors,
                time_steps=self.time_steps,
                forecast_horizon=self.forecast_horizon,
            )
            self.val_dataset = ForecastMambularDataset(
                cat_features_list=self.val_cat_tensors,
                num_features_list=self.val_num_tensors,
                time_steps=self.time_steps,
                forecast_horizon=self.forecast_horizon,
            )

    def _convert_to_tensor(self, data):
        """
        Helper method to convert preprocessed data into lists of categorical and numerical feature tensors.
        """
        cat_tensors = []
        num_tensors = []

        # Populate tensors for categorical features
        for key in self.cat_feature_info:
            cat_key = "cat_" + key
            if cat_key in data:
                cat_tensors.append(torch.tensor(data[cat_key], dtype=torch.long))

        # Populate tensors for numerical features
        for key in self.num_feature_info:
            num_key = "num_" + key
            if num_key in data:
                num_tensors.append(torch.tensor(data[num_key], dtype=torch.float32))

        return cat_tensors, num_tensors

    def train_dataloader(self):
        """
        Returns the training DataLoader with time series windowing.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Sequential order for time series
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        """
        Returns the validation DataLoader with time series windowing.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Sequential order for time series
            **self.dataloader_kwargs,
        )
