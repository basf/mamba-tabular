from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.ft_transformer import FTTransformer
from ..configs.fttransformer_config import DefaultFTTransformerConfig


class FTTransformerClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs
        )
