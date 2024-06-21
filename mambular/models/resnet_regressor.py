from .sklearn_base_regressor import SklearnBaseRegressor
from ..base_models.resnet import ResNet
from ..configs.resnet_config import DefaultResNetConfig


class ResNetRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)
