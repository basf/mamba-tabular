from ..base.sklearn_base_forecast import SklearnBaseTimeSeriesForecaster
from ...base_models.ft_transformer import FTTransformer
from ...configs.fttransformer_config import DefaultFTTransformerConfig


class FTTransformerForecast(SklearnBaseTimeSeriesForecaster):
    """
    RNN forecast. This class extends the SklearnBaseForecast class and uses the FTTransformer model
    with the default FTTransformer configuration.

    The accepted arguments to the FTTransformerForecast class include both the attributes in the DefaultFTTransformerConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    layer_sizes : list, default=(128, 128, 32)
        Sizes of the layers in the FTTransformer.
    activation : callable, default=nn.SELU()
        Activation function for the FTTransformer layers.
    skip_layers : bool, default=False
        Whether to skip layers in the FTTransformer.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the FTTransformer.
    skip_connections : bool, default=False
        Whether to use skip connections in the FTTransformer.
    batch_norm : bool, default=False
        Whether to use batch normalization in the FTTransformer layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the FTTransformer layers.
    use_embeddings : bool, default=False
        Whether to use embedding layers for all features.
    embedding_activation : callable, default=nn.Identity()
        Activation function for  embeddings.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    d_model : int, default=32
        Dimensionality of the embeddings.
    n_bins : int, default=50
        The number of bins to use for numerical feature binning. This parameter is relevant
        only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    numerical_preprocessing : str, default="ple"
        The preprocessing strategy for numerical features. Valid options are
        'binning', 'one_hot', 'standardization', and 'normalization'.
    use_decision_tree_bins : bool, default=False
        If True, uses decision tree regression/classification to determine
        optimal bin edges for numerical feature binning. This parameter is
        relevant only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    binning_strategy : str, default="uniform"
        Defines the strategy for binning numerical features. Options include 'uniform',
        'quantile', or other sklearn-compatible strategies.
    cat_cutoff : float or int, default=0.03
        Indicates the cutoff after which integer values are treated as categorical.
        If float, it's treated as a percentage. If int, it's the maximum number of
        unique values for a column to be considered categorical.
    treat_all_integers_as_numerical : bool, default=False
        If True, all integer columns will be treated as numerical, regardless
        of their unique value count or proportion.
    degree : int, default=3
        The degree of the polynomial features to be used in preprocessing.
    knots : int, default=12
        The number of knots to be used in spline transformations.
    """

    def __init__(self, **kwargs):
        super().__init__(
            model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs
        )

        self.config.forecast = True
        self.config.max_seq_length = kwargs.get("max_seq_length", 100)
        self.config.time_steps = kwargs.get("time_steps", 30)
        self.config.forecast_horizon = kwargs.get("forecast_horizon", 1)
