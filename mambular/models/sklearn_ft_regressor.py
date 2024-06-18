from .sklearn_base_regressor import SklearnBaseRegressor
from ..base_models.ft_transformer import FTTransformer
from ..utils.configs import DefaultFTTransformerConfig


class FTTransformerRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(
            model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs
        )
