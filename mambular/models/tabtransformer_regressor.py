from .sklearn_base_regressor import SklearnBaseRegressor
from ..base_models.tabtransformer import TabTransformer
from ..configs.tabtransformer_config import DefaultTabTransformerConfig


class TabTransformerRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(
            model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs
        )
