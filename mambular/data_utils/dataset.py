import numpy as np
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

        self.regression = regression
        if not self.regression:
            self.num_classes = len(np.unique(labels))
            if self.num_classes > 2:
                self.labels = labels.view(-1)
            else:
                self.num_classes = 1
                self.labels = labels
        else:
            self.labels = labels
            self.num_classes = 1

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
            torch.as_tensor(feature_tensor[idx]).clone(
            ).detach().to(torch.float32)
            for feature_tensor in self.num_features_list
        ]
        label = self.labels[idx]
        if self.regression:
            label = label.clone().detach().to(torch.float32)
        elif self.num_classes == 1:
            label = label.clone().detach().to(torch.float32)
        else:
            label = label.clone().detach().to(torch.long)

        # Keep categorical and numerical features separate
        return cat_features, num_features, label
