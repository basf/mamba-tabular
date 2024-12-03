from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_lss import SklearnBaseLSS
from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.cnn import CNN
from ..configs.cnn_config import DefaultCNNConfig


class CNNRegressor(SklearnBaseRegressor):
    """
    CNN regressor. This class extends the SklearnBaseRegressor class and uses the CNN model
    with the default CNN configuration.

    The accepted arguments to the CNNRegressor class include both the attributes in the DefaultCNNConfig dataclass
    and the parameters for the Preprocessor class.

    Optimizer Parameters
    --------------------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 regularization) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.

    Embedding Parameters
    ---------------------
    use_embeddings : bool, default=False
        Whether to use embedding layers for all features.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', 'plr', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    d_model : int, default=32
        Dimensionality of the embeddings.
    plr_lite : bool, default=False
        Whether to use a lightweight version of Piecewise Linear Regression (PLR).

    CNN Parameters
    --------------------
    input_channels : int, default=1
        Number of input channels (e.g., 1 for grayscale images).
    num_layers : int, default=4
        Number of convolutional layers.
    out_channels_list : list, default=(64, 64, 128, 128)
        List of output channels for each convolutional layer.
    kernel_size_list : list, default=(3, 3, 3, 3)
        List of kernel sizes for each convolutional layer.
    stride_list : list, default=(1, 1, 1, 1)
        List of stride values for each convolutional layer.
    padding_list : list, default=(1, 1, 1, 1)
        List of padding values for each convolutional layer.
    pooling_method : str, default="max"
        Pooling method ('max' or 'avg').
    pooling_kernel_size_list : list, default=(2, 2, 1, 1)
        List of kernel sizes for pooling layers for each convolutional layer.
    pooling_stride_list : list, default=(2, 2, 1, 1)
        List of stride values for pooling layers for each convolutional layer.

    Dropout Parameters
    -------------------
    dropout_rate : float, default=0.5
        Probability of dropping neurons during training.
    dropout_positions : list, default=None
        List of indices of layers after which dropout should be applied. If None, no dropout is applied.

    Preprocessing Params
    ---------------------
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
    - The accepted arguments to the CNNRegressor class are the same as the attributes in the DefaultCNNConfig dataclass.
    - CNNRegressor uses SklearnBaseRegressor as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseRegressor : The parent class for CNNRegressor.

    Examples
    --------
    >>> from mambular.models import CNNRegressor
    >>> model = CNNRegressor(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=CNN, config=DefaultCNNConfig, **kwargs)


class CNNLSS(SklearnBaseLSS):
    """
    CNN regressor. This class extends the SklearnBaseLSS class and uses the CNN model
    with the default CNN configuration.

    The accepted arguments to the CNNLSS class include both the attributes in the DefaultCNNConfig dataclass
    and the parameters for the Preprocessor class.

    Optimizer Parameters
    --------------------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 regularization) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.

    Embedding Parameters
    ---------------------
    use_embeddings : bool, default=False
        Whether to use embedding layers for all features.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', 'plr', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    d_model : int, default=32
        Dimensionality of the embeddings.
    plr_lite : bool, default=False
        Whether to use a lightweight version of Piecewise Linear Regression (PLR).

    CNN Parameters
    --------------------
    input_channels : int, default=1
        Number of input channels (e.g., 1 for grayscale images).
    num_layers : int, default=4
        Number of convolutional layers.
    out_channels_list : list, default=(64, 64, 128, 128)
        List of output channels for each convolutional layer.
    kernel_size_list : list, default=(3, 3, 3, 3)
        List of kernel sizes for each convolutional layer.
    stride_list : list, default=(1, 1, 1, 1)
        List of stride values for each convolutional layer.
    padding_list : list, default=(1, 1, 1, 1)
        List of padding values for each convolutional layer.
    pooling_method : str, default="max"
        Pooling method ('max' or 'avg').
    pooling_kernel_size_list : list, default=(2, 2, 1, 1)
        List of kernel sizes for pooling layers for each convolutional layer.
    pooling_stride_list : list, default=(2, 2, 1, 1)
        List of stride values for pooling layers for each convolutional layer.

    Dropout Parameters
    -------------------
    dropout_rate : float, default=0.5
        Probability of dropping neurons during training.
    dropout_positions : list, default=None
        List of indices of layers after which dropout should be applied. If None, no dropout is applied.

    Preprocessing Params
    ---------------------
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
    - The accepted arguments to the CNNLSS class are the same as the attributes in the DefaultCNNConfig dataclass.
    - CNNLSS uses SklearnBaseLSS as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseLSS : The parent class for CNNLSS.

    Examples
    --------
    >>> from mambular.models import CNNLSS
    >>> model = CNNLSS(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=CNN, config=DefaultCNNConfig, **kwargs)


class CNNClassifier(SklearnBaseClassifier):
    """
    CNN regressor. This class extends the SklearnBaseCLassifier class and uses the CNN model
    with the default CNN configuration.

    The accepted arguments to the CNNCLassifier class include both the attributes in the DefaultCNNConfig dataclass
    and the parameters for the Preprocessor class.

    Optimizer Parameters
    --------------------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 regularization) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.

    Embedding Parameters
    ---------------------
    use_embeddings : bool, default=False
        Whether to use embedding layers for all features.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', 'plr', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    d_model : int, default=32
        Dimensionality of the embeddings.
    plr_lite : bool, default=False
        Whether to use a lightweight version of Piecewise Linear Regression (PLR).

    CNN Parameters
    --------------------
    input_channels : int, default=1
        Number of input channels (e.g., 1 for grayscale images).
    num_layers : int, default=4
        Number of convolutional layers.
    out_channels_list : list, default=(64, 64, 128, 128)
        List of output channels for each convolutional layer.
    kernel_size_list : list, default=(3, 3, 3, 3)
        List of kernel sizes for each convolutional layer.
    stride_list : list, default=(1, 1, 1, 1)
        List of stride values for each convolutional layer.
    padding_list : list, default=(1, 1, 1, 1)
        List of padding values for each convolutional layer.
    pooling_method : str, default="max"
        Pooling method ('max' or 'avg').
    pooling_kernel_size_list : list, default=(2, 2, 1, 1)
        List of kernel sizes for pooling layers for each convolutional layer.
    pooling_stride_list : list, default=(2, 2, 1, 1)
        List of stride values for pooling layers for each convolutional layer.

    Dropout Parameters
    -------------------
    dropout_rate : float, default=0.5
        Probability of dropping neurons during training.
    dropout_positions : list, default=None
        List of indices of layers after which dropout should be applied. If None, no dropout is applied.

    Preprocessing Params
    ---------------------
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
    - The accepted arguments to the CNNCLassifier class are the same as the attributes in the DefaultCNNConfig dataclass.
    - CNNCLassifier uses SklearnBaseCLassifier as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseCLassifier : The parent class for CNNCLassifier.

    Examples
    --------
    >>> from mambular.models import CNNCLassifier
    >>> model = CNNCLassifier(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=CNN, config=DefaultCNNConfig, **kwargs)
