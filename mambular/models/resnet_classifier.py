from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.resnet import ResNet
from ..configs.resnet_config import DefaultResNetConfig


class ResNetClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)
