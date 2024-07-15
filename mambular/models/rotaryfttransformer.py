from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS

from ..base_models.rotary_fttransformer import RotaryFTTransformer
from ..configs.rotary_fttransformer_config import DefaultRotaryFTTransformerConfig


class RotaryFTTransformerRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(
            model=RotaryFTTransformer, config=DefaultRotaryFTTransformerConfig, **kwargs
        )


class RotaryFTTransformerClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            model=RotaryFTTransformer, config=DefaultRotaryFTTransformerConfig, **kwargs
        )


class RotaryFTTransformerLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(
            model=RotaryFTTransformer, config=DefaultRotaryFTTransformerConfig, **kwargs
        )
