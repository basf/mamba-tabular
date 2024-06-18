import torch
from torch.utils.data import Dataset


class MambularDataset(Dataset):
    """
    Custom dataset for handling structured data with separate categorical and numerical features, tailored for
    both regression and classification tasks.

    Parameters:
        cat_features_list (list of Tensors): A list of tensors representing the categorical features.
        num_features_list (list of Tensors): A list of tensors representing the numerical features.
        labels (Tensor): A tensor of labels.
        regression (bool, optional): A flag indicating if the dataset is for a regression task. Defaults to True.
    """

    def __init__(self, cat_features_list, num_features_list, labels, regression=True):
        self.cat_features_list = cat_features_list  # Categorical features tensors
        self.num_features_list = num_features_list  # Numerical features tensors
        self.labels = labels
        self.regression = regression

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the features and label for a given index.

        Parameters:
            idx (int): The index of the data point.

        Returns:
            tuple: A tuple containing two lists of tensors (one for categorical features and one for numerical
            features) and a single label (float if regression is True).
        """
        cat_features = [
            feature_tensor[idx] for feature_tensor in self.cat_features_list
        ]
        num_features = [
            torch.tensor(feature_tensor[idx], dtype=torch.float32)
            for feature_tensor in self.num_features_list
        ]
        label = self.labels[idx]
        if self.regression:
            # Convert the label to float for regression tasks
            # label = float(label)
            label = torch.tensor(label, dtype=torch.float32)

        else:
            label = torch.tensor(label, dtype=torch.long)

        # Keep categorical and numerical features separate
        return cat_features, num_features, label


class EmbeddingMambularDataset(Dataset):
    """
    A specialized version of MambularDataset intended for datasets related to protein studies, maintaining the
    same structure and functionality.

    This class is designed to handle structured data with separate categorical and numerical features, suitable
    for both regression and classification tasks within the context of protein studies.

    Parameters:
        cat_features_list (list of Tensors): A list of tensors representing the categorical features.
        num_features_list (list of Tensors): A list of tensors representing the numerical features.
        labels (Tensor): A tensor of labels.
        regression (bool, optional): A flag indicating if the dataset is for a regression task. Defaults to True.
    """

    def __init__(self, cat_features_list, num_features_list, labels, regression=True):
        self.cat_features_list = cat_features_list  # Categorical features tensors
        self.num_features_list = num_features_list  # Numerical features tensors
        self.labels = labels
        self.regression = regression

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the features and label for a given index in the context of protein studies.

        Parameters:
            idx (int): The index of the data point.

        Returns:
            tuple: A tuple containing two lists of tensors (one for categorical features and one for numerical
            features) and a single label (float if regression is True), specifically designed for protein data.
        """
        cat_features = [
            feature_tensor[idx] for feature_tensor in self.cat_features_list
        ]
        num_features = [
            torch.tensor(feature_tensor[idx], dtype=torch.float32)
            for feature_tensor in self.num_features_list
        ]
        label = self.labels[idx]
        if self.regression:
            # Convert the label to float for regression tasks
            # label = float(label)
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label, dtype=torch.long)

        # Keep categorical and numerical features separate
        return cat_features, num_features, label
