from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.resnet import ResNet
from ..configs.resnet_config import DefaultResNetConfig


class ResNetRegressor(SklearnBaseRegressor):
    """
    ResNet regressor. This class extends the SklearnBaseRegressor class and uses the ResNet model
    with the default ResNet configuration.

    The accepted arguments to the ResNetRegressor class include both the attributes in the DefaultResNetConfig dataclass
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
        Sizes of the layers in the ResNet.
    activation : callable, default=nn.SELU()
        Activation function for the ResNet layers.
    skip_layers : bool, default=False
        Whether to skip layers in the ResNet.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the ResNet.
    skip_connections : bool, default=True
        Whether to use skip connections in the ResNet.
    batch_norm : bool, default=True
        Whether to use batch normalization in the ResNet layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the ResNet layers.
    num_blocks : int, default=3
        Number of residual blocks in the ResNet.
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

    Notes
    -----
    - The accepted arguments to the ResNetRegressor class are the same as the attributes in the DefaultResNetConfig dataclass.
    - ResNetRegressor uses SklearnBaseRegressor as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseRegressor : The parent class for ResNetRegressor.

    Examples
    --------
    >>> from mambular.models import ResNetRegressor
    >>> model = ResNetRegressor()
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)


class ResNetClassifier(SklearnBaseClassifier):
    """
    ResNet classifier. This class extends the SklearnBaseClassifier class and uses the ResNet model
    with the default ResNet configuration.

    The accepted arguments to the ResNetClassifier class include both the attributes in the DefaultResNetConfig dataclass
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
        Sizes of the layers in the ResNet.
    activation : callable, default=nn.SELU()
        Activation function for the ResNet layers.
    skip_layers : bool, default=False
        Whether to skip layers in the ResNet.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the ResNet.
    skip_connections : bool, default=True
        Whether to use skip connections in the ResNet.
    batch_norm : bool, default=True
        Whether to use batch normalization in the ResNet layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the ResNet layers.
    num_blocks : int, default=3
        Number of residual blocks in the ResNet.
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

    Notes
    -----
    - The accepted arguments to the ResNetClassifier class are the same as the attributes in the DefaultResNetConfig dataclass.
    - ResNetClassifier uses SklearnBaseClassifier as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseClassifier : The parent class for ResNetClassifier.

    Examples
    --------
    >>> from mambular.models import ResNetClassifier
    >>> model = ResNetClassifier()
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)


class ResNetLSS(SklearnBaseLSS):
    """
    ResNet for distributional regression. This class extends the SklearnBaseLSS class and uses the ResNet model
    with the default ResNet configuration.

    The accepted arguments to the ResNetLSS class include both the attributes in the DefaultResNetConfig dataclass
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
        Sizes of the layers in the ResNet.
    activation : callable, default=nn.SELU()
        Activation function for the ResNet layers.
    skip_layers : bool, default=False
        Whether to skip layers in the ResNet.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    family : str, default=None
        Distributional family to be used for the model.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the ResNet.
    skip_connections : bool, default=True
        Whether to use skip connections in the ResNet.
    batch_norm : bool, default=True
        Whether to use batch normalization in the ResNet layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the ResNet layers.
    num_blocks : int, default=3
        Number of residual blocks in the ResNet.
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
    - The accepted arguments to the ResNetLSS class are the same as the attributes in the DefaultResNetConfig dataclass.
    - ResNetLSS uses SklearnBaseLSS as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseLSS : The parent class for ResNetLSS.

    Examples
    --------
    >>> from mambular.models import ResNetLSS
    >>> model = ResNetLSS()
    >>> model.fit(X_train, y_train, family="normal")
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)
