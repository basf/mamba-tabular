from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.ft_transformer import FTTransformer
from ..configs.fttransformer_config import DefaultFTTransformerConfig


class FTTransformerLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(
            model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs
        )
