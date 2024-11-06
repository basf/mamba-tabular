import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class ForecastMambularDataset(Dataset):
    """
    Enhanced dataset for handling time series forecasting with flexible options for additional temporal,
    contextual, and statistical features. Returns categorical and numerical features as separate lists and the targets
    as a tensor for easy handling in models expecting a specific structure.

    Parameters:
        cat_features_list (list of Tensors): A list of tensors representing the categorical features.
        num_features_list (list of Tensors): A list of tensors representing the numerical features.
        time_steps (int): Number of past observations to use as input.
        forecast_horizon (int): Number of future observations to predict.
        include_timestamps (bool): Whether to return timestamps for each sample.
        include_time_embeddings (bool): Whether to include cyclical time embeddings (e.g., day of the week).
        include_lagged_features (list of int): Lags to include as additional numerical features.
        include_rolling_stats (dict): Dict specifying rolling window sizes and statistics to compute.
        include_holiday_indicator (bool): Whether to add a holiday indicator feature.
        external_features (dict): External time-dependent features like weather or events, provided as tensors.
        scaling (str): Type of scaling to apply ("standard", "minmax", or None).
    """

    def __init__(
        self,
        cat_features_list,
        num_features_list,
        time_steps,
        forecast_horizon,
        timestamps=None,
        include_timestamps=False,
        include_time_embeddings=False,
        include_lagged_features=None,
        include_rolling_stats=None,
        include_holiday_indicator=False,
        holiday_list=None,
        external_features=None,
        scaling=None,
    ):
        self.cat_features_list = cat_features_list
        self.num_features_list = num_features_list
        self.time_steps = time_steps
        self.forecast_horizon = forecast_horizon
        self.timestamps = timestamps
        self.include_timestamps = include_timestamps
        self.include_time_embeddings = include_time_embeddings
        self.include_lagged_features = include_lagged_features or []
        self.include_rolling_stats = include_rolling_stats or {}
        self.include_holiday_indicator = include_holiday_indicator
        self.external_features = external_features or {}
        self.scaling = scaling
        self.holiday_list = holiday_list

    def __len__(self):
        return (
            len(self.num_features_list[0]) - self.time_steps - self.forecast_horizon + 1
        )

    def __getitem__(self, idx):
        cat_features_window = [
            feature[idx : idx + self.time_steps] for feature in self.cat_features_list
        ]
        num_features_window = [
            feature[idx : idx + self.time_steps] for feature in self.num_features_list
        ]

        # Target window (last entry in num_features is assumed to be the target series)
        target_window = self.num_features_list[-1][
            idx + self.time_steps : idx + self.time_steps + self.forecast_horizon
        ]

        # Include lagged features if specified
        if self.include_lagged_features:
            for lag in self.include_lagged_features:
                lagged_window = [
                    feature[idx - lag : idx + self.time_steps - lag]
                    for feature in self.num_features_list
                ]
                num_features_window.extend(lagged_window)

        # Include rolling statistics if specified
        if self.include_rolling_stats:
            for window_size, stat_funcs in self.include_rolling_stats.items():
                for func in stat_funcs:
                    rolling_window = [
                        func(
                            feature[
                                idx - window_size : idx + self.time_steps - window_size
                            ]
                        )
                        for feature in self.num_features_list
                    ]
                    num_features_window.extend(rolling_window)

        # Include time embeddings if specified
        if self.include_time_embeddings and self.timestamps is not None:
            time_embeddings = self._generate_time_embeddings(idx)
            num_features_window.append(time_embeddings)

        # Include holiday indicator if specified
        if self.include_holiday_indicator and self.timestamps is not None:
            holiday_indicator = self._generate_holiday_indicator(idx)
            num_features_window.append(holiday_indicator)

        # Include external features if provided
        for key, ext_feature in self.external_features.items():
            ext_window = ext_feature[idx : idx + self.time_steps]
            num_features_window.append(ext_window)

        # Apply scaling if specified
        if self.scaling:
            num_features_window = self._apply_scaling(num_features_window)

        return cat_features_window, num_features_window, target_window.squeeze(-1)

    def _generate_time_embeddings(self, idx):
        date = pd.to_datetime(self.timestamps[idx : idx + self.time_steps])
        day_of_week = np.sin(2 * np.pi * date.dayofweek / 7)
        month_of_year = np.sin(2 * np.pi * date.month / 12)
        return torch.tensor(
            np.vstack([day_of_week, month_of_year]).T, dtype=torch.float
        )

    def _generate_holiday_indicator(self, idx):
        date = pd.to_datetime(self.timestamps[idx : idx + self.time_steps])
        holidays = pd.Series(
            [1 if d in self.holidays_list else 0 for d in date]
        )  # Use a predefined holiday list
        return torch.tensor(holidays.values, dtype=torch.float)

    def _apply_scaling(self, features):
        # Apply scaling on a feature-by-feature basis
        scaled_features = []
        for feature in features:
            if self.scaling == "standard":
                mean, std = feature.mean(), feature.std()
                scaled_feature = (feature - mean) / (std + 1e-8)
            elif self.scaling == "minmax":
                min_val, max_val = feature.min(), feature.max()
                scaled_feature = (feature - min_val) / (max_val - min_val + 1e-8)
            else:
                scaled_feature = feature
            scaled_features.append(scaled_feature)
        return scaled_features
