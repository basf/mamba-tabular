from .sklearn_base_regressor import SklearnBaseRegressor
from ..base_models.mlp import MLP
from ..configs.mlp_config import DefaultMLPConfig


class MLPRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=MLP, config=DefaultMLPConfig, **kwargs)
