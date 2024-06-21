from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.mambular import Mambular
from ..configs.mambular_config import DefaultMambularConfig


class MambularClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)
