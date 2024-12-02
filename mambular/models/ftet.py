from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS

from ..base_models.ftet import FTET
from ..configs.ftet_config import DefaultFTETConfig


class FTETRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=FTET, config=DefaultFTETConfig, **kwargs)


class FTETClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=FTET, config=DefaultFTETConfig, **kwargs)


class FTETLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=FTET, config=DefaultFTETConfig, **kwargs)
