from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_classifier import SklearnBaseClassifier
from .sklearn_base_lss import SklearnBaseLSS
from ..base_models.node import NODE
from ..configs.node_config import DefaultNODEConfig


class NODERegressor(SklearnBaseRegressor):
    """
    Neural Oblivious Decision Ensemble (NODE) Regressor. Slightly different with a MLP as a tabular task specific head. This class extends the SklearnBaseRegressor class and uses the NODE model
    with the default NODE configuration.

    The accepted arguments to the NODERegressor class include both the attributes in the DefaultNODEConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-4.
    lr_patience : int, optional
        Number of epochs without improvement after which the learning rate will be reduced. Default is 10.
    weight_decay : float, optional
        Weight decay (L2 regularization penalty) applied by the optimizer. Default is 1e-6.
    lr_factor : float, optional
        Factor by which the learning rate is reduced when there is no improvement. Default is 0.1.
    norm : str, optional
        Type of normalization to use. Default is None.
    use_embeddings : bool, optional
        Whether to use embedding layers for categorical features. Default is False.
    embedding_activation : callable, optional
        Activation function to apply to embeddings. Default is `nn.Identity`.
    layer_norm_after_embedding : bool, optional
        Whether to apply layer normalization after embedding layers. Default is False.
    d_model : int, optional
        Dimensionality of the embedding space. Default is 32.
    num_layers : int, optional
        Number of dense layers in the model. Default is 4.
    layer_dim : int, optional
        Dimensionality of each dense layer. Default is 128.
    tree_dim : int, optional
        Dimensionality of the output from each tree leaf. Default is 1.
    depth : int, optional
        Depth of each decision tree in the ensemble. Default is 6.
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
    - The accepted arguments to the NODERegressor class are the same as the attributes in the DefaultNODEConfig dataclass.
    - NODERegressor uses SklearnBaseRegressor as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseRegressor : The parent class for NODERegressor.

    Examples
    --------
    >>> from mambular.models import NODERegressor
    >>> model = NODERegressor(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=NODE, config=DefaultNODEConfig, **kwargs)


class NODEClassifier(SklearnBaseClassifier):
    """
    Neural Oblivious Decision Ensemble (NODE) Classifier. Slightly different with a MLP as a tabular task specific head. This class extends the SklearnBaseClassifier class and uses the NODE model
    with the default NODE configuration.

    The accepted arguments to the NODEClassifier class include both the attributes in the DefaultNODEConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-4.
    lr_patience : int, optional
        Number of epochs without improvement after which the learning rate will be reduced. Default is 10.
    weight_decay : float, optional
        Weight decay (L2 regularization penalty) applied by the optimizer. Default is 1e-6.
    lr_factor : float, optional
        Factor by which the learning rate is reduced when there is no improvement. Default is 0.1.
    norm : str, optional
        Type of normalization to use. Default is None.
    use_embeddings : bool, optional
        Whether to use embedding layers for categorical features. Default is False.
    embedding_activation : callable, optional
        Activation function to apply to embeddings. Default is `nn.Identity`.
    layer_norm_after_embedding : bool, optional
        Whether to apply layer normalization after embedding layers. Default is False.
    d_model : int, optional
        Dimensionality of the embedding space. Default is 32.
    num_layers : int, optional
        Number of dense layers in the model. Default is 4.
    layer_dim : int, optional
        Dimensionality of each dense layer. Default is 128.
    tree_dim : int, optional
        Dimensionality of the output from each tree leaf. Default is 1.
    depth : int, optional
        Depth of each decision tree in the ensemble. Default is 6.
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
    - The accepted arguments to the NODEClassifier class are the same as the attributes in the DefaultNODEConfig dataclass.
    - NODEClassifier uses SklearnBaseClassifieras the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseClassifier : The parent class for NODEClassifier.

    Examples
    --------
    >>> from mambular.models import NODEClassifier
    >>> model = NODEClassifier(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=NODE, config=DefaultNODEConfig, **kwargs)


class NODELSS(SklearnBaseLSS):
    """
    Neural Oblivious Decision Ensemble (NODE) for disrtibutional regression. Slightly different with a MLP as a tabular task specific head. This class extends the SklearnBaseLSS class and uses the NODE model
    with the default NODE configuration.

    The accepted arguments to the NODELSS class include both the attributes in the DefaultNODEConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-4.
    lr_patience : int, optional
        Number of epochs without improvement after which the learning rate will be reduced. Default is 10.
    weight_decay : float, optional
        Weight decay (L2 regularization penalty) applied by the optimizer. Default is 1e-6.
    lr_factor : float, optional
        Factor by which the learning rate is reduced when there is no improvement. Default is 0.1.
    norm : str, optional
        Type of normalization to use. Default is None.
    use_embeddings : bool, optional
        Whether to use embedding layers for categorical features. Default is False.
    embedding_activation : callable, optional
        Activation function to apply to embeddings. Default is `nn.Identity`.
    layer_norm_after_embedding : bool, optional
        Whether to apply layer normalization after embedding layers. Default is False.
    d_model : int, optional
        Dimensionality of the embedding space. Default is 32.
    num_layers : int, optional
        Number of dense layers in the model. Default is 4.
    layer_dim : int, optional
        Dimensionality of each dense layer. Default is 128.
    tree_dim : int, optional
        Dimensionality of the output from each tree leaf. Default is 1.
    depth : int, optional
        Depth of each decision tree in the ensemble. Default is 6.
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
    - The accepted arguments to the NODELSS class are the same as the attributes in the DefaultNODEConfig dataclass.
    - NODELSS uses SklearnBaseLSS as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseLSS : The parent class for NODELSS.

    Examples
    --------
    >>> from mambular.models import NODELSS
    >>> model = NODELSS(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=NODE, config=DefaultNODEConfig, **kwargs)
