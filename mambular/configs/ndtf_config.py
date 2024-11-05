from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultNDTFConfig:
    """
    Configuration class for the default Neural Decision Tree Forest (NDTF) model with predefined hyperparameters.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) applied to the model's weights during optimization.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced when a plateau is reached.
    min_depth : int, default=2
        Minimum depth of trees in the forest. Controls the simplest model structure.
    max_depth : int, default=10
        Maximum depth of trees in the forest. Controls the maximum complexity of the trees.
    temperature : float, default=0.1
        Temperature parameter for softening the node decisions during path probability calculation.
    node_sampling : float, default=0.3
        Fraction of nodes sampled for regularization penalty calculation. Reduces computation by focusing on a subset of nodes.
    lamda : float, default=0.3
        Regularization parameter to control the complexity of the paths, penalizing overconfident or imbalanced paths.
    n_ensembles : int, default=12
        Number of trees in the forest
    penalty_factor : float, default=0.01
        Factor with which the penalty is multiplied
    """

    lr: float = 1e-4
    lr_patience: int = 5
    weight_decay: float = 1e-7
    lr_factor: float = 0.1
    min_depth: int = 4
    max_depth: int = 16
    temperature: float = 0.1
    node_sampling: float = 0.3
    lamda: float = 0.3
    n_ensembles: int = 12
    penalty_factor: float = 1e-08
