from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.resnet import ResNet
from ..configs.resnet_config import DefaultResNetConfig


class ResNetRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)


class ResNetClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)


class ResNetLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)
