from dataclasses import dataclass, asdict, field
import json
import os
import math
from typing import Union, Type, List
from .normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
import torch.nn as nn


@dataclass
class MambularConfig:
    """
    A configuration class specific to the Mambular model.
    Handles Mamba-specific hyperparameters as well as vocabulary size and output dimensions.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
        n_layers (int): The number of MambaBlocks in the model.
        dt_rank (Union[int, str]): The rank of the dynamical time tensor.
            Can be an integer or 'auto' to calculate automatically based on d_model.
        d_state (int): The dimensionality of the state tensor.
        expand_factor (int): The factor by which the inner dimensionality is expanded.
        d_conv (int): The dimensionality of the convolutional layer.

        dt_min (float): The minimum value for dynamical time.
        dt_max (float): The maximum value for dynamical time.
        dt_init (str): The initialization method for dynamical time. Either 'constant' or 'random'.
        dt_scale (float): The scale factor for dynamical time initialization.
        dt_init_floor (float): The floor value for dynamical time initialization.

        dropout (float): The dropout probability.
        bias (bool): Whether to include bias in linear layers.
        weight_decay (float): weight decay in optimizer.
        conv_bias (bool): Whether to include bias in the convolutional layer.
        vocab_size (list): The sizes of the vocabulary for the features used by the Mambular model.
        output_dimension (int): The dimensionality of the output layer.
        pooling_method (str): The pooling method for combining token embeddings.
            Options: 'avg', 'max', 'sum', 'cls_token'.
        norm (nn.Module): The normalization layer to use.
            Options: RMSNorm, LayerNorm, LearnableLayerScaling, BatchNorm, InstanceNorm, GroupNorm.

    Methods:
        __post_init__(): Performs additional initialization steps or checks after instance creation.
        save_pretrained(save_directory: str): Saves the configuration to a JSON file.

    Raises:
        ValueError: If invalid values are provided for pooling method or normalization layer.
    """

    VALID_POOLING_METHODS = ["avg", "max", "sum", "cls_token"]

    VALID_NORMALIZATION_LAYERS = {
        "RMSNorm": RMSNorm,
        "LayerNorm": LayerNorm,
        "LearnableLayerScaling": LearnableLayerScaling,
        "BatchNorm": BatchNorm,
        "InstanceNorm": InstanceNorm,
        "GroupNorm": GroupNorm,
    }

    d_model: int = 64
    n_layers: int = 6
    dt_rank: Union[int, str] = "auto"
    d_state: int = 32
    expand_factor: int = 2
    d_conv: int = 8

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    dropout: float = 0.05

    bias: bool = False
    weight_decay: float = 0.025
    conv_bias: bool = True
    output_dimension: int = 1
    pooling_method: str = "avg"
    norm: Union[str, Type[nn.Module]] = RMSNorm
    num_embedding_activation: str = "linear"
    tabular_head_units: list = field(default_factory=lambda: [128, 64, 64])
    tabular_head_activation: str = "relu"
    tabular_head_dropout: float = 0.3
    layer_norm_after_embedding: bool = True

    def __post_init__(self):
        """
        Called automatically after the initialization of MambularConfig instances.
        Performs additional initialization steps or checks, if required.
        """
        self.d_inner = self.expand_factor * self.d_model

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        # Check if the provided pooling method is valid
        if self.pooling_method not in self.VALID_POOLING_METHODS:
            raise ValueError(
                f"Invalid pooling method: {self.pooling_method}. "
                f"Valid options are: {', '.join(self.VALID_POOLING_METHODS)}"
            )

        # Check if the provided normalization layer is valid
        if (
            isinstance(self.norm, type)
            and self.norm.__name__ not in self.VALID_NORMALIZATION_LAYERS
        ):
            raise ValueError(
                f"Invalid normalization layer: {self.norm.__name__}. "
                f"Valid options are: {', '.join(self.VALID_NORMALIZATION_LAYERS.keys())}"
            )
        elif (
            isinstance(self.norm, str)
            and self.norm not in self.VALID_NORMALIZATION_LAYERS
        ):
            raise ValueError(
                f"Invalid normalization layer: {self.norm}. "
                f"Valid options are: {', '.join(self.VALID_NORMALIZATION_LAYERS.keys())}"
            )

    def save_pretrained(self, save_directory: str):
        """
        Saves the configuration parameters of the MambularConfig instance to a JSON file
        in the specified directory. This is useful for model persistence, reproducibility,
        or reloading the model configuration in the future.

        Parameters:
            save_directory (str): The directory path where the configuration JSON file will be saved.

        Returns:
            None: The method prints the path where the configuration is saved but does not return any value.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Define the configuration file path
        config_file = os.path.join(save_directory, "config.json")

        # Convert the dataclass to a dictionary and then to a JSON string
        config_dict = asdict(self)
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=4)

        print(f"Configuration saved in {config_file}")
