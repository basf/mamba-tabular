from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.tabtransformer import TabTransformer
from ..configs.tabtransformer_config import DefaultTabTransformerConfig


class TabTransformerRegressor(SklearnBaseRegressor):
    """
    TabTransformer regressor. This class extends the SklearnBaseRegressor class and uses the TabTransformer model
    with the default TabTransformer configuration.

    The accepted arguments to the TabTransformerRegressor class include both the attributes in the DefaultTabTransformerConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    family : str, default=None
        Distributional family to be used for the model.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=8
        Number of layers in the transformer.
    n_heads : int, default=4
        Number of attention heads in the transformer.
    attn_dropout : float, default=0.3
        Dropout rate for the attention mechanism.
    ff_dropout : float, default=0.3
        Dropout rate for the feed-forward layers.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the transformer.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    head_layer_sizes : list, default=(128, 64, 32)
        Sizes of the layers in the head of the model.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    pooling_method : str, default="avg"
        Pooling method to be used ('cls', 'avg', etc.).
    norm_first : bool, default=True
        Whether to apply normalization before other operations in each transformer block.
    bias : bool, default=True
        Whether to use bias in the linear layers.
    transformer_activation : callable, default=nn.SELU()
        Activation function for the transformer layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    transformer_dim_feedforward : int, default=512
        Dimensionality of the feed-forward layers in the transformer.
    cat_encoding : str, default="int"
        whether to use integer encoding or one-hot encoding for cat features.
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

    Notes
    -----
    - The accepted arguments to the TabTransformerRegressor class are the same as the attributes in the DefaultTabTransformerConfig dataclass.
    - TabTransformerRegressor uses SklearnBaseRegressor as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseRegressor : The parent class for TabTransformerRegressor.

    Examples
    --------
    >>> from mambular.models import TabTransformerRegressor
    >>> model = TabTransformerRegressor(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(
            model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs
        )


class TabTransformerClassifier(SklearnBaseClassifier):
    """
    TabTransformer Classifier. This class extends the SklearnBaseClassifier class and uses the TabTransformer model
    with the default TabTransformer configuration.

    The accepted arguments to the TabTransformerClassifier class include both the attributes in the DefaultTabTransformerConfig dataclass
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
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=8
        Number of layers in the transformer.
    n_heads : int, default=4
        Number of attention heads in the transformer.
    attn_dropout : float, default=0.3
        Dropout rate for the attention mechanism.
    ff_dropout : float, default=0.3
        Dropout rate for the feed-forward layers.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the transformer.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    head_layer_sizes : list, default=(128, 64, 32)
        Sizes of the layers in the head of the model.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    pooling_method : str, default="avg"
        Pooling method to be used ('cls', 'avg', etc.).
    norm_first : bool, default=True
        Whether to apply normalization before other operations in each transformer block.
    bias : bool, default=True
        Whether to use bias in the linear layers.
    transformer_activation : callable, default=nn.SELU()
        Activation function for the transformer layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    transformer_dim_feedforward : int, default=512
        Dimensionality of the feed-forward layers in the transformer.
    cat_encoding : str, default="int"
        whether to use integer encoding or one-hot encoding for cat features.
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

    Notes
    -----
    - The accepted arguments to the TabTransformerClassifier class are the same as the attributes in the DefaultTabTransformerConfig dataclass.
    - TabTransformerClassifier uses SklearnBaseClassifier as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseClassifier : The parent class for TabTransformerClassifier.

    Examples
    --------
    >>> from mambular.models import TabTransformerClassifier
    >>> model = TabTransformerClassifier(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(
            model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs
        )


class TabTransformerLSS(SklearnBaseLSS):
    """
    TabTransformer for distributional regression. This class extends the SklearnBaseLSS class and uses the TabTransformer model
    with the default TabTransformer configuration.

    The accepted arguments to the TabTransformerLSS class include both the attributes in the DefaultTabTransformerConfig dataclass
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
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=8
        Number of layers in the transformer.
    n_heads : int, default=4
        Number of attention heads in the transformer.
    attn_dropout : float, default=0.3
        Dropout rate for the attention mechanism.
    ff_dropout : float, default=0.3
        Dropout rate for the feed-forward layers.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the transformer.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    head_layer_sizes : list, default=(128, 64, 32)
        Sizes of the layers in the head of the model.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    pooling_method : str, default="avg"
        Pooling method to be used ('cls', 'avg', etc.).
    norm_first : bool, default=True
        Whether to apply normalization before other operations in each transformer block.
    bias : bool, default=True
        Whether to use bias in the linear layers.
    transformer_activation : callable, default=nn.SELU()
        Activation function for the transformer layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    transformer_dim_feedforward : int, default=512
        Dimensionality of the feed-forward layers in the transformer.
    cat_encoding : str, default="int"
        whether to use integer encoding or one-hot encoding for cat features.
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
    task : str, default="regression"
        Indicates the type of machine learning task ('regression' or 'classification'). This can
        influence certain preprocessing behaviors, especially when using decision tree-based binning as ple.
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

    Notes
    -----
    - The accepted arguments to the TabTransformerLSS class are the same as the attributes in the DefaultTabTransformerConfig dataclass.
    - TabTransformerLSS uses SklearnBaseLSS as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseLSS : The parent class for TabTransformerLSS.

    Examples
    --------
    >>> from mambular.models import TabTransformerLSS
    >>> model = TabTransformerLSS(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train, family="normal")
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(
            model=TabTransformer, config=DefaultTabTransformerConfig, **kwargs
        )
