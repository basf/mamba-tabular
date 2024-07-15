from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS

from ..base_models.rnn import RNN
from ..configs.rnn_config import DefaultRNNConfig


class RNNRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=RNN, config=DefaultRNNConfig, **kwargs)


class RNNClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(model=RNN, config=DefaultRNNConfig, **kwargs)


class RNNLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=RNN, config=DefaultRNNConfig, **kwargs)
