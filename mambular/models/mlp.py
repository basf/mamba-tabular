from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.mlp import MLP
from ..configs.mlp_config import DefaultMLPConfig


class MLPRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=MLP, config=DefaultMLPConfig, **kwargs)


class MLPClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=MLP, config=DefaultMLPConfig, **kwargs)


class MLPLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=MLP, config=DefaultMLPConfig, **kwargs)
