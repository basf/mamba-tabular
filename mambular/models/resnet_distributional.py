from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.resnet import ResNet
from ..configs.resnet_config import DefaultResNetConfig


class ResNetLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)
