from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.mambular import Mambular
from ..configs.mambular_config import DefaultMambularConfig


class MambularLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)
