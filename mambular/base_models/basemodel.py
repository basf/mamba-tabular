import torch
import torch.nn as nn
import os
import logging


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        """
        Initializes the BaseModel with given hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Hyperparameters to be saved and used in the model.
        """
        super(BaseModel, self).__init__()
        self.hparams = kwargs

    def save_hyperparameters(self, ignore=[]):
        """
        Saves the hyperparameters while ignoring specified keys.

        Parameters
        ----------
        ignore : list, optional
            List of keys to ignore while saving hyperparameters, by default [].
        """
        self.hparams = {k: v for k, v in self.hparams.items() if k not in ignore}
        for key, value in self.hparams.items():
            setattr(self, key, value)

    def save_model(self, path):
        """
        Save the model parameters to the given path.

        Parameters
        ----------
        path : str
            Path to save the model parameters.
        """
        torch.save(self.state_dict(), path)
        print(f"Model parameters saved to {path}")

    def load_model(self, path, device="cpu"):
        """
        Load the model parameters from the given path.

        Parameters
        ----------
        path : str
            Path to load the model parameters from.
        device : str, optional
            Device to map the model parameters, by default 'cpu'.
        """
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f"Model parameters loaded from {path}")

    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.

        Returns
        -------
        int
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_parameters(self):
        """
        Freeze the model parameters by setting `requires_grad` to False.
        """
        for param in self.parameters():
            param.requires_grad = False
        print("All model parameters have been frozen.")

    def unfreeze_parameters(self):
        """
        Unfreeze the model parameters by setting `requires_grad` to True.
        """
        for param in self.parameters():
            param.requires_grad = True
        print("All model parameters have been unfrozen.")

    def log_parameters(self, logger=None):
        """
        Log the hyperparameters and model parameters.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance to log the parameters, by default None.
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.info("Hyperparameters:")
        for key, value in self.hparams.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Total number of trainable parameters: {self.count_parameters()}")

    def parameter_count(self):
        """
        Get a dictionary of parameter counts for each layer in the model.

        Returns
        -------
        dict
            Dictionary where keys are layer names and values are parameter counts.
        """
        param_count = {}
        for name, param in self.named_parameters():
            param_count[name] = param.numel()
        return param_count

    def get_device(self):
        """
        Get the device on which the model is located.

        Returns
        -------
        torch.device
            Device on which the model is located.
        """
        return next(self.parameters()).device

    def to_device(self, device):
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : torch.device or str
            Device to move the model to.
        """
        self.to(device)
        print(f"Model moved to {device}")

    def print_summary(self):
        """
        Print a summary of the model, including the architecture and parameter counts.
        """
        print(self)
        print(f"\nTotal number of trainable parameters: {self.count_parameters()}")
        print("\nParameter counts by layer:")
        for name, count in self.parameter_count().items():
            print(f"  {name}: {count}")
