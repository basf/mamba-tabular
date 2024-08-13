from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_lss import SklearnBaseLSS
from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.mambatab import MambaTab
from ..configs.mambatab_config import DefaultMambaTabConfig


class MambaTabRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=MambaTab, config=DefaultMambaTabConfig, **kwargs)


class MambaTabClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=MambaTab, config=DefaultMambaTabConfig, **kwargs)


class MambaTabLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=MambaTab, config=DefaultMambaTabConfig, **kwargs)
