from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.tabtransformer import TabTransformer
from ..configs.tabtransformer_config import DefaultTabTransformerConfig


class TabTransformerClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs
        )
