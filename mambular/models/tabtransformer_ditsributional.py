from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.tabtransformer import TabTransformer
from ..configs.tabtransformer_config import DefaultTabTransformerConfig


class TabTransformerLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(
            model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs
        )
