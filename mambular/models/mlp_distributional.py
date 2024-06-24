from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.mlp import MLP
from ..configs.mlp_config import DefaultMLPConfig


class MLPLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=MLP, config=DefaultMLPConfig, **kwargs)
