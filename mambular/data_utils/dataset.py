import numpy as np
import torch
from torch.utils.data import Dataset


import numpy as np
import torch
from torch.utils.data import Dataset


class MambularDataset(Dataset):
    """Custom dataset for handling structured data with separate categorical and
    numerical features, tailored for both regression and classification tasks.

    Parameters
    ----------
        cat_features_list (list of Tensors): A list of tensors representing the categorical features.
        num_features_list (list of Tensors): A list of tensors representing the numerical features.
        labels (Tensor, optional): A tensor of labels. If None, the dataset is used for prediction.
        regression (bool, optional): A flag indicating if the dataset is for a regression task. Defaults to True.
    """

    def __init__(
        self, cat_features_list, num_features_list, labels=None, regression=True
    ):
        self.cat_features_list = cat_features_list  # Categorical features tensors
        self.num_features_list = num_features_list  # Numerical features tensors
        self.regression = regression

        if labels is not None:
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
        else:
            self.labels = None  # No labels in prediction mode

    def __len__(self):
        return len(self.num_features_list[0])  # Use numerical features length

    def __getitem__(self, idx):
        """Retrieves the features and label for a given index.

        Parameters
        ----------
            idx (int): The index of the data point.

        Returns
        -------
            tuple: A tuple containing two lists of tensors (one for categorical features and one for numerical features)
            and a single label (if available).
        """
        cat_features = [
            feature_tensor[idx] for feature_tensor in self.cat_features_list
        ]
        num_features = [
            torch.as_tensor(feature_tensor[idx]).clone().detach().to(torch.float32)
            for feature_tensor in self.num_features_list
        ]

        if self.labels is not None:
            label = self.labels[idx]
            if self.regression:
                label = label.clone().detach().to(torch.float32)
            elif self.num_classes == 1:
                label = label.clone().detach().to(torch.float32)
            else:
                label = label.clone().detach().to(torch.long)
            return num_features, cat_features, label
        else:
            return num_features, cat_features  # No label in prediction mode
