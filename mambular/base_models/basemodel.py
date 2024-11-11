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

    def initialize_pooling_layers(self, config, n_inputs):
        """
        Initializes the layers needed for learnable pooling methods based on self.pooling_method.
        """
        if self.pooling_method == "learned_flatten":
            # Flattening + Linear layer
            self.learned_flatten_pooling = nn.Linear(
                n_inputs * config.dim_feedforward, config.dim_feedforward
            )

        elif self.pooling_method == "attention":
            # Attention-based pooling with learnable attention weights
            self.attention_weights = nn.Parameter(torch.randn(config.dim_feedforward))

        elif self.pooling_method == "gated":
            # Gated pooling with a learned gating layer
            self.gate_layer = nn.Linear(config.dim_feedforward, config.dim_feedforward)

        elif self.pooling_method == "rnn":
            # RNN-based pooling: Use a small RNN (e.g., LSTM)
            self.pooling_rnn = nn.LSTM(
                input_size=config.dim_feedforward,
                hidden_size=config.dim_feedforward,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )

        elif self.pooling_method == "conv":
            # Conv1D-based pooling with global max pooling
            self.conv1d_pooling = nn.Conv1d(
                in_channels=config.dim_feedforward,
                out_channels=config.dim_feedforward,
                kernel_size=3,  # or a configurable kernel size
                padding=1,  # ensures output has the same sequence length
            )

    def pool_sequence(self, out):
        """
        Pools the sequence dimension based on self.pooling_method.
        """

        if self.pooling_method == "avg":
            return out.mean(
                dim=1
            )  # Shape: (batch_size, ensemble_size, hidden_size) or (batch_size, hidden_size)
        elif self.pooling_method == "max":
            return out.max(dim=1)[0]
        elif self.pooling_method == "sum":
            return out.sum(dim=1)
        elif self.pooling_method == "last":
            return out[:, -1, :]
        elif self.pooling_method == "cls":
            return out[:, 0, :]
        elif self.pooling_method == "learned_flatten":
            # Flatten sequence and apply a learned linear layer
            batch_size, seq_len, hidden_size = out.shape
            out = out.reshape(
                batch_size, -1
            )  # Shape: (batch_size, seq_len * hidden_size)
            return self.learned_flatten_pooling(out)  # Shape: (batch_size, hidden_size)
        elif self.pooling_method == "attention":
            # Attention-based pooling
            attention_scores = torch.einsum(
                "bsh,h->bs", out, self.attention_weights
            )  # Shape: (batch_size, seq_len)
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(
                -1
            )  # Shape: (batch_size, seq_len, 1)
            out = (out * attention_weights).sum(
                dim=1
            )  # Weighted sum across the sequence, Shape: (batch_size, hidden_size)
            return out
        elif self.pooling_method == "gated":
            # Gated pooling
            gates = torch.sigmoid(
                self.gate_layer(out)
            )  # Shape: (batch_size, seq_len, hidden_size)
            out = (out * gates).sum(dim=1)  # Shape: (batch_size, hidden_size)
            return out
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling_method}")
