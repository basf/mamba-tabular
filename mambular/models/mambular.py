from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_lss import SklearnBaseLSS
from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.mambular import Mambular
from ..configs.mambular_config import DefaultMambularConfig


class MambularRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)


class MambularClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)


class MambularLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)
