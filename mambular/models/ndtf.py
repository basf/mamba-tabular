from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.ndtf import NDTF
from ..configs.ndtf_config import DefaultNDTFConfig


class NDTFRegressor(SklearnBaseRegressor):
    """
    Multi-Layer Perceptron regressor. This class extends the SklearnBaseRegressor class and uses the NDTF model
    with the default NDTF configuration.

    The accepted arguments to the NDTFRegressor class include both the attributes in the DefaultNDTFConfig dataclass
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
    min_depth : int, default=2
        Minimum depth of trees in the forest. Controls the simplest model structure.
    max_depth : int, default=10
        Maximum depth of trees in the forest. Controls the maximum complexity of the trees.
    temperature : float, default=0.1
        Temperature parameter for softening the node decisions during path probability calculation.
    node_sampling : float, default=0.3
        Fraction of nodes sampled for regularization penalty calculation. Reduces computation by focusing on a subset of nodes.
    lamda : float, default=0.3
        Regularization parameter to control the complexity of the paths, penalizing overconfident or imbalanced paths.
    n_ensembles : int, default=12
        Number of trees in the forest
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
    - The accepted arguments to the NDTFRegressor class are the same as the attributes in the DefaultNDTFConfig dataclass.
    - NDTFRegressor uses SklearnBaseRegressor as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseRegressor : The parent class for NDTFRegressor.

    Examples
    --------
    >>> from mambular.models import NDTFRegressor
    >>> model = NDTFRegressor(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=NDTF, config=DefaultNDTFConfig, **kwargs)


class NDTFClassifier(SklearnBaseClassifier):
    """
    Multi-Layer Perceptron classifier. This class extends the SklearnBaseClassifier class and uses the NDTF model
    with the default NDTF configuration.

    The accepted arguments to the NDTFClassifier class include both the attributes in the DefaultNDTFConfig dataclass
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
    min_depth : int, default=2
        Minimum depth of trees in the forest. Controls the simplest model structure.
    max_depth : int, default=10
        Maximum depth of trees in the forest. Controls the maximum complexity of the trees.
    temperature : float, default=0.1
        Temperature parameter for softening the node decisions during path probability calculation.
    node_sampling : float, default=0.3
        Fraction of nodes sampled for regularization penalty calculation. Reduces computation by focusing on a subset of nodes.
    lamda : float, default=0.3
        Regularization parameter to control the complexity of the paths, penalizing overconfident or imbalanced paths.
    n_ensembles : int, default=12
        Number of trees in the forest
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
    - The accepted arguments to the NDTFClassifier class are the same as the attributes in the DefaultNDTFConfig dataclass.
    - NDTFClassifier uses SklearnBaseClassifieras the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseRegressor : The parent class for NDTFClassifier.

    Examples
    --------
    >>> from mambular.models import NDTFClassifier
    >>> model = NDTFClassifier(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=NDTF, config=DefaultNDTFConfig, **kwargs)


class NDTFLSS(SklearnBaseLSS):
    """
    Multi-Layer Perceptron for distributional regression. This class extends the SklearnBaseLSS class and uses the NDTF model
    with the default NDTF configuration.

    The accepted arguments to the NDTFLSS class include both the attributes in the DefaultNDTFConfig dataclass
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
    min_depth : int, default=2
        Minimum depth of trees in the forest. Controls the simplest model structure.
    max_depth : int, default=10
        Maximum depth of trees in the forest. Controls the maximum complexity of the trees.
    temperature : float, default=0.1
        Temperature parameter for softening the node decisions during path probability calculation.
    node_sampling : float, default=0.3
        Fraction of nodes sampled for regularization penalty calculation. Reduces computation by focusing on a subset of nodes.
    lamda : float, default=0.3
        Regularization parameter to control the complexity of the paths, penalizing overconfident or imbalanced paths.
    n_ensembles : int, default=12
        Number of trees in the forest
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
    - The accepted arguments to the NDTFLSS class are the same as the attributes in the DefaultNDTFConfig dataclass.
    - NDTFLSS uses SklearnBaseLSS as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseLSS : The parent class for NDTFLSS.

    Examples
    --------
    >>> from mambular.models import NDTFLSS
    >>> model = NDTFLSS(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=NDTF, config=DefaultNDTFConfig, **kwargs)
