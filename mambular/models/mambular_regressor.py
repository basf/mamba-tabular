from .sklearn_base_regressor import SklearnBaseRegressor
from ..base_models.mambular import Mambular
from ..configs.mambular_config import DefaultMambularConfig


class MambularRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)
