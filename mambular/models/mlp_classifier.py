from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.mlp import MLP
from ..configs.mlp_config import DefaultMLPConfig


class MLPClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=MLP, config=DefaultMLPConfig, **kwargs)
