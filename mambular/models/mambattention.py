from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_lss import SklearnBaseLSS
from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.mambattn import MambAttention
from ..configs.mambattention_config import DefaultMambAttentionConfig


class MambAttentionRegressor(SklearnBaseRegressor):
    """
    MambAttention regressor. This class extends the SklearnBaseRegressor class and uses the MambAttention model
    with the default MambAttention configuration.

    The accepted arguments to the MambAttentionRegressor class include both the attributes in the DefaultMambAttentionConfig dataclass
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
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    d_conv : int, default=16
        Dimensionality of the convolutional layers.
    conv_bias : bool, default=True
        Whether to use bias in the convolutional layers.
    dropout : float, default=0.05
        Dropout rate for regularization.
    dt_rank : str, default="auto"
        Rank of the decision tree.
    d_state : int, default=32
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree.
    dt_init : str, default="random"
        Initialization method for decision tree.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the model.
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
        Pooling method to be used ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to use bidirectional processing of the input sequences.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through mamba blocks.
    use_cls : bool, default=True
        Whether to append a cls to the end of each 'sequence'.
    shuffle_embeddings : bool, default=False.
        Whether to shuffle the embeddings before being passed to the Mamba layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    AD_weight_decay : bool, default=True
        whether weight decay is also applied to A-D matrices.
    BC_layer_norm: bool, default=False
        whether to apply layer normalization to B-C matrices.
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
    - The accepted arguments to the MambAttentionRegressor class are the same as the attributes in the DefaultMambAttentionConfig dataclass.
    - MambAttentionRegressor uses SklearnBaseRegressor as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    MambAttention.models.SklearnBaseRegressor : The parent class for MambAttentionRegressor.

    Examples
    --------
    >>> from MambAttention.models import MambAttentionRegressor
    >>> model = MambAttentionRegressor(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(
            model=MambAttention, config=DefaultMambAttentionConfig, **kwargs
        )


class MambAttentionClassifier(SklearnBaseClassifier):
    """
    MambAttention classifier. This class extends the SklearnBaseClassifier class and uses the MambAttention model
    with the default MambAttention configuration.

    The accepted arguments to the MambAttentionClassifier class include both the attributes in the DefaultMambAttentionConfig dataclass
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
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    d_conv : int, default=16
        Dimensionality of the convolutional layers.
    conv_bias : bool, default=True
        Whether to use bias in the convolutional layers.
    dropout : float, default=0.05
        Dropout rate for regularization.
    dt_rank : str, default="auto"
        Rank of the decision tree.
    d_state : int, default=32
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree.
    dt_init : str, default="random"
        Initialization method for decision tree.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the model.
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
        Pooling method to be used ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to use bidirectional processing of the input sequences.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through mamba blocks.
    shuffle_embeddings : bool, default=False.
        Whether to shuffle the embeddings before being passed to the Mamba layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    AD_weight_decay : bool, default=True
        whether weight decay is also applied to A-D matrices.
    BC_layer_norm: bool, default=False
        whether to apply layer normalization to B-C matrices.
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
    - The accepted arguments to the MambAttentionClassifier class are the same as the attributes in the DefaultMambAttentionConfig dataclass.
    - MambAttentionClassifier uses SklearnBaseClassifier as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    MambAttention.models.SklearnBaseClassifier : The parent class for MambAttentionClassifier.

    Examples
    --------
    >>> from MambAttention.models import MambAttentionClassifier
    >>> model = MambAttentionClassifier(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(
            model=MambAttention, config=DefaultMambAttentionConfig, **kwargs
        )


class MambAttentionLSS(SklearnBaseLSS):
    """
    MambAttention for distributional regression. This class extends the SklearnBaseLSS class and uses the MambAttention model
    with the default MambAttention configuration.

    The accepted arguments to the MambAttentionLSS class include both the attributes in the DefaultMambAttentionConfig dataclass
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
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    d_conv : int, default=16
        Dimensionality of the convolutional layers.
    conv_bias : bool, default=True
        Whether to use bias in the convolutional layers.
    dropout : float, default=0.05
        Dropout rate for regularization.
    dt_rank : str, default="auto"
        Rank of the decision tree.
    d_state : int, default=32
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree.
    dt_init : str, default="random"
        Initialization method for decision tree.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the model.
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
        Pooling method to be used ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to use bidirectional processing of the input sequences.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through mamba blocks.
    n_bins : int, default=50
        The number of bins to use for numerical feature binning. This parameter is relevant
        only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    shuffle_embeddings : bool, default=False.
        Whether to shuffle the embeddings before being passed to the Mamba layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    AD_weight_decay : bool, default=True
        whether weight decay is also applied to A-D matrices.
    BC_layer_norm: bool, default=False
        whether to apply layer normalization to B-C matrices.
    cat_encoding : str, default="int"
        whether to use integer encoding or one-hot encoding for cat features.
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
    - The accepted arguments to the MambAttentionLSS class are the same as the attributes in the DefaultMambAttentionConfig dataclass.
    - MambAttentionLSS uses SklearnBaseLSS as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    MambAttention.models.SklearnBaseLSS : The parent class for MambAttentionLSS.

    Examples
    --------
    >>> from MambAttention.models import MambAttentionLSS
    >>> model = MambAttentionLSS(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train, family="normal")
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(
            model=MambAttention, config=DefaultMambAttentionConfig, **kwargs
        )
