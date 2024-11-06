import torch
from torch.utils.data import Dataset


class ForecastMambularDataset(Dataset):
    """
    Custom dataset for handling univariate or multivariate time series forecasting with separate categorical
    and numerical features. Generates input-output pairs based on a sliding window approach.

    Parameters:
        cat_features_list (list of Tensors): A list of tensors representing the categorical features.
        num_features_list (list of Tensors): A list of tensors representing the numerical features.
        time_steps (int): Number of past observations to use as input.
        forecast_horizon (int): Number of future observations to predict.
    """

    def __init__(
        self, cat_features_list, num_features_list, time_steps, forecast_horizon
    ):
        self.cat_features_list = (
            cat_features_list  # List of categorical feature tensors
        )
        self.num_features_list = num_features_list  # List of numerical feature tensors
        self.time_steps = time_steps
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return (
            len(self.num_features_list[0]) - self.time_steps - self.forecast_horizon + 1
        )

    def __getitem__(self, idx):
        """
        Retrieves a sliding window of data for time series forecasting, including separate lists for categorical
        and numerical features and a target window.

        Parameters:
            idx (int): Index of the starting point of the window.

        Returns:
            tuple: A tuple containing a list of categorical feature tensors, a list of numerical feature tensors,
            and a tensor of future targets.
        """
        # Extract sliding windows for categorical features
        cat_features_window = [
            feature_tensor[idx : idx + self.time_steps]
            for feature_tensor in self.cat_features_list
        ]

        # Extract sliding windows for numerical features
        num_features_window = [
            feature_tensor[idx : idx + self.time_steps]
            for feature_tensor in self.num_features_list
        ]

        # Extract target window from the numerical feature tensor, as it’s assumed to contain the target series
        # Here we assume that the last element in num_features_list represents the target values
        target_window = self.num_features_list[-1][
            idx + self.time_steps : idx + self.time_steps + self.forecast_horizon
        ]

        return cat_features_window, num_features_window, target_window
