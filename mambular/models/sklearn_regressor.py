from .sklearn_base_regressor import SklearnBaseRegressor
from ..base_models.mambular_base import Mambular
from ..utils.configs import DefaultMambularConfig


class MambularRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)
