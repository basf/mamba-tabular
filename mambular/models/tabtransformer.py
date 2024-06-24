from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.tabtransformer import TabTransformer
from ..configs.tabtransformer_config import DefaultTabTransformerConfig


class TabTransformerRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(
            model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs
        )


class TabTransformerClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs
        )


class TabTransformerLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(
            model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs
        )
