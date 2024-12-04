from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_lss import SklearnBaseLSS
from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.mambular import Mambular
from ..configs.mambular_config import DefaultMambularConfig


class MambularRegressor(SklearnBaseRegressor):
    """
    Mambular regressor. This class extends the SklearnBaseRegressor class and uses the Mambular model
    with the default Mambular configuration.

    The accepted arguments to the MambularRegressor class include both the attributes in the DefaultMambularConfig dataclass
    and the parameters for the Preprocessor class.

     Optimizer Parameters
    --------------------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.

    Mambular Model Parameters
    -----------------------
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=4
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    dropout : float, default=0.0
        Dropout rate for regularization.
    dt_rank : str, default="auto"
        Rank of the decision tree used in the model.
    d_state : int, default=128
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree parameters.
    dt_init : str, default="random"
        Initialization method for decision tree parameters.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="LayerNorm"
        Type of normalization used ('LayerNorm', 'RMSNorm', etc.).
    activation : callable, default=nn.SiLU()
        Activation function for the model.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    AD_weight_decay : bool, default=True
        Whether weight decay is applied to A-D matrices.
    BC_layer_norm : bool, default=False
        Whether to apply layer normalization to B-C matrices.

    Embedding Parameters
    ---------------------
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    shuffle_embeddings : bool, default=False
        Whether to shuffle embeddings before being passed to Mamba layers.
    cat_encoding : str, default="int"
        Encoding method for categorical features ('int', 'one-hot', etc.).

    Head Parameters
    ---------------
    head_layer_sizes : list, default=()
        Sizes of the layers in the model's head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.

    Additional Features
    --------------------
    pooling_method : str, default="avg"
        Pooling method to use ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to process data bidirectionally.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through Mamba blocks.
    use_cls : bool, default=False
        Whether to append a CLS token to the input sequences.
    use_pscan : bool, default=False
        Whether to use PSCAN for the state-space model.

    Mamba Version
    -------------
    mamba_version : str, default="mamba-torch"
        Version of the Mamba model to use ('mamba-torch', 'mamba1', 'mamba2').

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
    - The accepted arguments to the MambularRegressor class are the same as the attributes in the DefaultMambularConfig dataclass.
    - MambularRegressor uses SklearnBaseRegressor as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseRegressor : The parent class for MambularRegressor.

    Examples
    --------
    >>> from mambular.models import MambularRegressor
    >>> model = MambularRegressor(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)


class MambularClassifier(SklearnBaseClassifier):
    """
    Mambular classifier. This class extends the SklearnBaseClassifier class and uses the Mambular model
    with the default Mambular configuration.

    The accepted arguments to the MambularClassifier class include both the attributes in the DefaultMambularConfig dataclass
    and the parameters for the Preprocessor class.

    Mambular Model Parameters
    -----------------------
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=4
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    dropout : float, default=0.0
        Dropout rate for regularization.
    dt_rank : str, default="auto"
        Rank of the decision tree used in the model.
    d_state : int, default=128
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree parameters.
    dt_init : str, default="random"
        Initialization method for decision tree parameters.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="LayerNorm"
        Type of normalization used ('LayerNorm', 'RMSNorm', etc.).
    activation : callable, default=nn.SiLU()
        Activation function for the model.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    AD_weight_decay : bool, default=True
        Whether weight decay is applied to A-D matrices.
    BC_layer_norm : bool, default=False
        Whether to apply layer normalization to B-C matrices.

    Embedding Parameters
    ---------------------
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    shuffle_embeddings : bool, default=False
        Whether to shuffle embeddings before being passed to Mamba layers.
    cat_encoding : str, default="int"
        Encoding method for categorical features ('int', 'one-hot', etc.).

    Head Parameters
    ---------------
    head_layer_sizes : list, default=()
        Sizes of the layers in the model's head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.

    Additional Features
    --------------------
    pooling_method : str, default="avg"
        Pooling method to use ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to process data bidirectionally.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through Mamba blocks.
    use_cls : bool, default=False
        Whether to append a CLS token to the input sequences.
    use_pscan : bool, default=False
        Whether to use PSCAN for the state-space model.

    Mamba Version
    -------------
    mamba_version : str, default="mamba-torch"
        Version of the Mamba model to use ('mamba-torch', 'mamba1', 'mamba2').

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
    - The accepted arguments to the MambularClassifier class are the same as the attributes in the DefaultMambularConfig dataclass.
    - MambularClassifier uses SklearnBaseClassifier as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseClassifier : The parent class for MambularClassifier.

    Examples
    --------
    >>> from mambular.models import MambularClassifier
    >>> model = MambularClassifier(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)


class MambularLSS(SklearnBaseLSS):
    """
    Mambular for distributional regression. This class extends the SklearnBaseLSS class and uses the Mambular model
    with the default Mambular configuration.

    The accepted arguments to the MambularLSS class include both the attributes in the DefaultMambularConfig dataclass
    and the parameters for the Preprocessor class.

    Mambular Model Parameters
    -----------------------
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=4
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    dropout : float, default=0.0
        Dropout rate for regularization.
    dt_rank : str, default="auto"
        Rank of the decision tree used in the model.
    d_state : int, default=128
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree parameters.
    dt_init : str, default="random"
        Initialization method for decision tree parameters.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="LayerNorm"
        Type of normalization used ('LayerNorm', 'RMSNorm', etc.).
    activation : callable, default=nn.SiLU()
        Activation function for the model.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    AD_weight_decay : bool, default=True
        Whether weight decay is applied to A-D matrices.
    BC_layer_norm : bool, default=False
        Whether to apply layer normalization to B-C matrices.

    Embedding Parameters
    ---------------------
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    shuffle_embeddings : bool, default=False
        Whether to shuffle embeddings before being passed to Mamba layers.
    cat_encoding : str, default="int"
        Encoding method for categorical features ('int', 'one-hot', etc.).

    Head Parameters
    ---------------
    head_layer_sizes : list, default=()
        Sizes of the layers in the model's head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.

    Additional Features
    --------------------
    pooling_method : str, default="avg"
        Pooling method to use ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to process data bidirectionally.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through Mamba blocks.
    use_cls : bool, default=False
        Whether to append a CLS token to the input sequences.
    use_pscan : bool, default=False
        Whether to use PSCAN for the state-space model.

    Mamba Version
    -------------
    mamba_version : str, default="mamba-torch"
        Version of the Mamba model to use ('mamba-torch', 'mamba1', 'mamba2').

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
    - The accepted arguments to the MambularLSS class are the same as the attributes in the DefaultMambularConfig dataclass.
    - MambularLSS uses SklearnBaseLSS as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseLSS : The parent class for MambularLSS.

    Examples
    --------
    >>> from mambular.models import MambularLSS
    >>> model = MambularLSS(d_model=64, n_layers=8)
    >>> model.fit(X_train, y_train, family="normal")
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)
