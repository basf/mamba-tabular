from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS

from ..base_models.ft_transformer import FTTransformer
from ..configs.fttransformer_config import DefaultFTTransformerConfig


class FTTransformerRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(
            model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs
        )


class FTTransformerClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs
        )


class FTTransformerLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(
            model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs
        )
