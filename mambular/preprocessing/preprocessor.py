import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .basis_expansion import RBFExpansion, SigmoidExpansion, SplineExpansion
from .ple_encoding import PLE
from .prepro_utils import (
    ContinuousOrdinalEncoder,
    CustomBinner,
    LanguageEmbeddingTransformer,
    NoTransformer,
    OneHotFromOrdinal,
    ToFloatTransformer,
)
from sklearn.base import TransformerMixin


class Preprocessor(TransformerMixin):
    """A comprehensive preprocessor for structured data, capable of handling both numerical and categorical features.
    It supports various preprocessing strategies for numerical data, including binning, one-hot encoding,
    standardization,and minmax. Categorical features can be transformed using continuous ordinal encoding.
    Additionally, it allows for the use of decision tree-derived bin edges for numerical feature binning.

    The class is designed to work seamlessly with pandas DataFrames, facilitating easy integration into
    machine learning pipelines.

    Parameters
    ----------
    feature_preprocessing: dict or None
            Dictionary mapping column names to preprocessing techniques. Example:
            {
                "num_feature1": "minmax",
                "num_feature2": "ple",
                "cat_feature1": "one-hot",
                "cat_feature2": "int"
            }
    n_bins : int, default=50
        The number of bins to use for numerical feature binning. This parameter is relevant
        only if `numerical_preprocessing` is set to 'binning', 'ple' or 'one-hot'.
    numerical_preprocessing : str, default="ple"
        The preprocessing strategy for numerical features. Valid options are
        'ple', 'binning', 'one-hot', 'standardization', 'min-max', 'quantile', 'polynomial', 'robust', 'rbf', 'sigmoid'.
        'splines', 'box-cox', 'yeo-johnson' and None
    categorical_preprocessing : str, default="int"
        The preprocessing strategy for categorical features. Valid options are
        'int', 'one-hot', None
    use_decision_tree_bins : bool, default=False
        If True, uses decision tree regression/classification to determine
        optimal bin edges for numerical feature binning. This parameter is
        relevant only if `numerical_preprocessing` is set to 'binning' or 'one-hot'.
    binning_strategy : str, default="uniform"
        Defines the strategy for binning numerical features. Options include 'uniform',
        'quantile', or other sklearn-compatible strategies.
    task : str, default="regression"
        Indicates the type of machine learning task ('regression' or 'classification'). This can
        influence certain preprocessing behaviors, especially when using decision tree-based binning.
    cat_cutoff : float or int, default=0.03
        Indicates the cutoff after which integer values are treated as categorical.
        If float, it's treated as a percentage. If int, it's the maximum number of
        unique values for a column to be considered categorical.
    treat_all_integers_as_numerical : bool, default=False
        If True, all integer columns will be treated as numerical, regardless
        of their unique value count or proportion.
    scaling_strategy : str, default="minmax"
        The scaling strategy to use for numerical features before applying PLE, Splines, RBF or Sigmoid.
        Options include 'standardization', 'minmax', 'none'.
    degree : int, default=3
        The degree of the polynomial features to be used in preprocessing. It also affects the degree of
        splines if splines are used.
    n_knots : int, default=12
        The number of knots to be used in spline transformations.
    use_decision_tree_knots : bool, default=True
        If True, uses decision tree regression to determine optimal knot positions for splines.
    knots_strategy : str, default="quantile"
        Defines the strategy for determining knot positions in spline transformations
        if `use_decision_tree_knots` is False. Options include 'uniform', 'quantile'.
    spline_implementation : str, default="sklearn"
        The library to use for spline implementation. Options include 'scipy' and 'sklearn'.

    Attributes
    ----------
    column_transformer : ColumnTransformer
        An instance of sklearn's ColumnTransformer that holds the
        configured preprocessing pipelines for different feature types.
    fitted : bool
        Indicates whether the preprocessor has been fitted to the data.
    """

    def __init__(
        self,
        feature_preprocessing=None,
        n_bins=64,
        numerical_preprocessing="ple",
        categorical_preprocessing="int",
        use_decision_tree_bins=False,
        binning_strategy="uniform",
        task="regression",
        cat_cutoff=0.03,
        treat_all_integers_as_numerical=False,
        degree=3,
        scaling_strategy="minmax",
        n_knots=64,
        use_decision_tree_knots=True,
        knots_strategy="uniform",
        spline_implementation="sklearn",
    ):
        self.n_bins = n_bins
        self.numerical_preprocessing = (
            numerical_preprocessing.lower()
            if numerical_preprocessing is not None
            else "none"
        )
        self.categorical_preprocessing = (
            categorical_preprocessing.lower()
            if categorical_preprocessing is not None
            else "none"
        )
        if self.numerical_preprocessing not in [
            "ple",
            "binning",
            "one-hot",
            "standardization",
            "minmax",
            "quantile",
            "polynomial",
            "robust",
            "splines",
            "box-cox",
            "yeo-johnson",
            "rbf",
            "sigmoid",
            "none",
        ]:
            raise ValueError(
                "Invalid numerical_preprocessing value. Supported values are 'ple', 'binning', 'box-cox', \
                'one-hot', 'standardization', 'quantile', 'polynomial', 'splines', 'minmax' , 'robust',\
                      'rbf', 'sigmoid', or 'None'."
            )

        if self.categorical_preprocessing not in [
            "int",
            "one-hot",
            "pretrained",
            "none",
        ]:
            raise ValueError(
                "invalid categorical_preprocessing value. Supported values are 'int', 'pretrained', 'none' and 'one-hot'"
            )

        self.use_decision_tree_bins = use_decision_tree_bins
        self.feature_preprocessing = feature_preprocessing or {}
        self.column_transformer = None
        self.fitted = False
        self.binning_strategy = binning_strategy
        self.task = task
        self.cat_cutoff = cat_cutoff
        self.treat_all_integers_as_numerical = treat_all_integers_as_numerical
        self.degree = degree
        self.scaling_strategy = scaling_strategy
        self.n_knots = n_knots
        self.use_decision_tree_knots = use_decision_tree_knots
        self.knots_strategy = knots_strategy
        self.spline_implementation = spline_implementation

    def get_params(self, deep=True):
        """Get parameters for the preprocessor.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return parameters of subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            "n_bins": self.n_bins,
            "numerical_preprocessing": self.numerical_preprocessing,
            "categorical_preprocessing": self.categorical_preprocessing,
            "use_decision_tree_bins": self.use_decision_tree_bins,
            "binning_strategy": self.binning_strategy,
            "task": self.task,
            "cat_cutoff": self.cat_cutoff,
            "treat_all_integers_as_numerical": self.treat_all_integers_as_numerical,
            "degree": self.degree,
            "scaling_strategy": self.scaling_strategy,
            "n_knots": self.n_knots,
            "use_decision_tree_knots": self.use_decision_tree_knots,
            "knots_strategy": self.knots_strategy,
        }
        return params

    def set_params(self, **params):
        """Set parameters for the preprocessor.

        Parameters
        ----------
        **params : dict
            Parameter names mapped to their new values.

        Returns
        -------
        self : object
            Preprocessor instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _detect_column_types(self, X):
        """Identifies and separates the features in the dataset into numerical and categorical types based on the data
        type and the proportion of unique values.

        Parameters
        ----------
            X (DataFrame or dict): The input dataset, where the features are columns in a DataFrame or keys in a dict.


        Returns
        -------
            tuple: A tuple containing two lists, the first with the names of numerical features and the
            second with the names of categorical features.
        """
        categorical_features = []
        numerical_features = []

        if isinstance(X, dict):
            X = pd.DataFrame(X)

        for col in X.columns:
            num_unique_values = X[col].nunique()
            total_samples = len(X[col])

            if self.treat_all_integers_as_numerical and X[col].dtype.kind == "i":
                numerical_features.append(col)
            else:
                if isinstance(self.cat_cutoff, float):
                    cutoff_condition = (
                        num_unique_values / total_samples
                    ) < self.cat_cutoff
                elif isinstance(self.cat_cutoff, int):
                    cutoff_condition = num_unique_values < self.cat_cutoff
                else:
                    raise ValueError(
                        "cat_cutoff should be either a float or an integer."
                    )

                if X[col].dtype.kind not in "iufc" or (
                    X[col].dtype.kind == "i" and cutoff_condition
                ):
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)

        return numerical_features, categorical_features

    def _fit_embeddings(self, embeddings):
        if embeddings is not None:
            self.embeddings = True
            self.embedding_dimensions = {}
            if isinstance(embeddings, np.ndarray):
                self.embedding_dimensions["embeddings_1"] = embeddings.shape[1]
            elif isinstance(embeddings, list) and all(
                isinstance(e, np.ndarray) for e in embeddings
            ):
                for idx, e in enumerate(embeddings):
                    self.embedding_dimensions[f"embedding_{idx + 1}"] = e.shape[1]
        else:
            self.embeddings = False

    def fit(self, X, y=None, embeddings=None):
        """Fits the preprocessor to the data by identifying feature types and configuring the appropriate
        transformations for each feature. It sets up a column transformer with a pipeline of transformations for
        numerical and categorical features based on the specified preprocessing strategy.

        Parameters
        ----------
            X (DataFrame or dict): The input dataset to fit the preprocessor on.
            y (array-like, optional): The target variable. Required if `use_decision_tree_bins` is True
            for determining optimal bin edges using decision trees.


        Returns
        -------
            self: The fitted Preprocessor instance.
        """
        if isinstance(X, dict):
            X = pd.DataFrame(X)

        self._fit_embeddings(embeddings)

        numerical_features, categorical_features = self._detect_column_types(X)
        transformers = []

        if numerical_features:
            for feature in numerical_features:
                feature_preprocessing = self.feature_preprocessing.get(
                    feature, self.numerical_preprocessing
                )

                # extended the annotation list if new transformer is added, either from sklearn or custom
                numeric_transformer_steps: list[
                    tuple[
                        str,
                        SimpleImputer
                        | StandardScaler
                        | MinMaxScaler
                        | QuantileTransformer
                        | PolynomialFeatures
                        | RobustScaler
                        | SplineTransformer
                        | KBinsDiscretizer
                        | CustomBinner
                        | OneHotFromOrdinal
                        | PLE
                        | PowerTransformer
                        | NoTransformer
                        | SplineExpansion
                        | RBFExpansion
                        | SigmoidExpansion,
                    ]
                ] = [("imputer", SimpleImputer(strategy="mean"))]
                if feature_preprocessing in ["binning", "one-hot"]:
                    bins = (
                        self._get_decision_tree_bins(X[[feature]], y, [feature])
                        if self.use_decision_tree_bins
                        else self.n_bins
                    )

                    if isinstance(bins, int):
                        numeric_transformer_steps.extend(
                            [
                                (
                                    "discretizer",
                                    KBinsDiscretizer(
                                        n_bins=(
                                            bins
                                            if isinstance(bins, int)
                                            else len(bins) - 1
                                        ),
                                        encode="ordinal",
                                        strategy=self.binning_strategy,  # type: ignore
                                        subsample=200_000 if len(X) > 200_000 else None,
                                    ),
                                ),
                            ]
                        )
                    else:
                        numeric_transformer_steps.extend(
                            [("CustomBinner", CustomBinner(bins=bins[0]))]
                        )

                    if feature_preprocessing == "one-hot":
                        numeric_transformer_steps.extend(
                            [
                                ("onehot_from_ordinal", OneHotFromOrdinal()),
                            ]
                        )

                elif feature_preprocessing == "standardization":
                    numeric_transformer_steps.append(("scaler", StandardScaler()))

                elif feature_preprocessing == "minmax":
                    numeric_transformer_steps.append(
                        ("minmax", MinMaxScaler(feature_range=(-1, 1)))
                    )

                elif feature_preprocessing == "quantile":
                    numeric_transformer_steps.append(
                        (
                            "quantile",
                            QuantileTransformer(
                                n_quantiles=self.n_bins, random_state=101
                            ),
                        )
                    )

                elif feature_preprocessing == "polynomial":
                    if self.scaling_strategy == "standardization":
                        numeric_transformer_steps.append(("scaler", StandardScaler()))
                    elif self.scaling_strategy == "minmax":
                        numeric_transformer_steps.append(
                            ("minmax", MinMaxScaler(feature_range=(-1, 1)))
                        )
                    numeric_transformer_steps.append(
                        (
                            "polynomial",
                            PolynomialFeatures(self.degree, include_bias=False),
                        )
                    )

                elif feature_preprocessing == "robust":
                    numeric_transformer_steps.append(("robust", RobustScaler()))

                elif feature_preprocessing == "splines":
                    if self.scaling_strategy == "standardization":
                        numeric_transformer_steps.append(("scaler", StandardScaler()))
                    elif self.scaling_strategy == "minmax":
                        numeric_transformer_steps.append(
                            ("minmax", MinMaxScaler(feature_range=(-1, 1)))
                        )
                    numeric_transformer_steps.append(
                        (
                            "splines",
                            SplineExpansion(
                                n_knots=self.n_knots,
                                degree=self.degree,
                                use_decision_tree=self.use_decision_tree_knots,
                                task=self.task,
                                strategy=self.knots_strategy,
                                spline_implementation=self.spline_implementation,
                            ),
                        ),
                    )

                elif feature_preprocessing == "rbf":
                    if self.scaling_strategy == "standardization":
                        numeric_transformer_steps.append(("scaler", StandardScaler()))
                    elif self.scaling_strategy == "minmax":
                        numeric_transformer_steps.append(
                            ("minmax", MinMaxScaler(feature_range=(-1, 1)))
                        )
                    numeric_transformer_steps.append(
                        (
                            "rbf",
                            RBFExpansion(
                                n_centers=self.n_knots,
                                use_decision_tree=self.use_decision_tree_knots,
                                strategy=self.knots_strategy,
                                task=self.task,
                            ),
                        )
                    )

                elif feature_preprocessing == "sigmoid":
                    if self.scaling_strategy == "standardization":
                        numeric_transformer_steps.append(("scaler", StandardScaler()))
                    elif self.scaling_strategy == "minmax":
                        numeric_transformer_steps.append(
                            ("minmax", MinMaxScaler(feature_range=(-1, 1)))
                        )
                    numeric_transformer_steps.append(
                        (
                            "sigmoid",
                            SigmoidExpansion(
                                n_centers=self.n_knots,
                                use_decision_tree=self.use_decision_tree_knots,
                                strategy=self.knots_strategy,
                                task=self.task,
                            ),
                        )
                    )

                elif feature_preprocessing == "ple":
                    numeric_transformer_steps.append(
                        ("minmax", MinMaxScaler(feature_range=(-1, 1)))
                    )
                    numeric_transformer_steps.append(
                        ("ple", PLE(n_bins=self.n_bins, task=self.task))
                    )

                elif feature_preprocessing == "box-cox":
                    numeric_transformer_steps.append(
                        ("minmax", MinMaxScaler(feature_range=(1e-03, 1)))  # type: ignore
                    )
                    numeric_transformer_steps.append(
                        ("check_positive", MinMaxScaler(feature_range=(1e-3, 1)))  # type: ignore
                    )
                    numeric_transformer_steps.append(
                        (
                            "box-cox",
                            PowerTransformer(method="box-cox", standardize=True),
                        )
                    )

                elif feature_preprocessing == "yeo-johnson":
                    numeric_transformer_steps.append(
                        (
                            "yeo-johnson",
                            PowerTransformer(method="yeo-johnson", standardize=True),
                        )
                    )

                elif feature_preprocessing == "none":
                    numeric_transformer_steps.append(
                        (
                            "none",
                            NoTransformer(),
                        )
                    )

                numeric_transformer = Pipeline(numeric_transformer_steps)

                transformers.append((f"num_{feature}", numeric_transformer, [feature]))

        if categorical_features:
            for feature in categorical_features:
                feature_preprocessing = self.feature_preprocessing.get(
                    feature, self.categorical_preprocessing
                )
                if feature_preprocessing == "int":
                    # Use ContinuousOrdinalEncoder for "int"
                    categorical_transformer = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("continuous_ordinal", ContinuousOrdinalEncoder()),
                        ]
                    )
                elif feature_preprocessing == "one-hot":
                    # Use OneHotEncoder for "one-hot"
                    categorical_transformer = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder()),
                            ("to_float", ToFloatTransformer()),
                        ]
                    )

                elif feature_preprocessing == "none":
                    # Use OneHotEncoder for "one-hot"
                    categorical_transformer = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("none", NoTransformer()),
                        ]
                    )
                elif feature_preprocessing == "pretrained":
                    categorical_transformer = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("pretrained", LanguageEmbeddingTransformer()),
                        ]
                    )
                else:
                    raise ValueError(
                        f"Unknown categorical_preprocessing type: {feature_preprocessing}"
                    )

                # Append the transformer for the current categorical feature
                transformers.append(
                    (f"cat_{feature}", categorical_transformer, [feature])
                )

        self.column_transformer = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )
        self.column_transformer.fit(X, y)

        self.fitted = True

    def _get_decision_tree_bins(self, X, y, numerical_features):
        """Uses decision tree models to determine optimal bin edges for numerical feature binning. This method is used
        when `use_decision_tree_bins` is True.

        Parameters
        ----------
            X (DataFrame): The input dataset containing only the numerical features for which the bin
            edges are to be determined.
            y (array-like): The target variable for training the decision tree models.
            numerical_features (list of str): The names of the numerical features for which the bin edges
            are to be determined.


        Returns
        -------
            list: A list of arrays, where each array contains the bin edges determined by the decision tree
            for a numerical feature.
        """
        bins = []
        for feature in numerical_features:
            tree_model = (
                DecisionTreeClassifier(max_depth=5)
                if y.dtype.kind in "bi"
                else DecisionTreeRegressor(max_depth=5)
            )
            tree_model.fit(X[[feature]], y)
            thresholds = tree_model.tree_.threshold[tree_model.tree_.feature != -2]  # type: ignore
            bin_edges = np.sort(np.unique(thresholds))

            bins.append(
                np.concatenate(([X[feature].min()], bin_edges, [X[feature].max()]))
            )
        return bins

    def transform(self, X, embeddings=None):
        """Transforms the input data using the preconfigured column transformer and converts the output into a
        dictionary format with keys corresponding to transformed feature names and values as arrays of transformed data.

        This method converts the sparse or dense matrix returned by the column transformer into a more accessible
        dictionary format, where each key-value pair represents a feature and its transformed data.
        Transforms the input data using the preconfigured column transformer and converts the output into a dictionary
        format with keys corresponding to transformed feature names and values as arrays of transformed data.

        This method converts the sparse or dense matrix returned by the column transformer into a more accessible
        dictionary format, where each key-value pair represents a feature and its transformed data.

        Parameters
        ----------
            X (DataFrame): The input data to be transformed.
            embeddings (np.array or list of np.arrays, optional): The embedding data to include in the transformation.

        Returns
        -------
            dict: A dictionary where keys are the names of the features (as per the transformations defined in the
            column transformer) and the values are numpy arrays of the transformed data.
        """
        if not self.fitted:
            raise NotFittedError(
                "The preprocessor must be fitted before transforming new data. Use .fit or .fit_transform"
            )
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        else:
            X = X.copy()
        transformed_X = self.column_transformer.transform(X)  # type: ignore

        # Now let's convert this into a dictionary of arrays, one per column
        transformed_dict = self._split_transformed_output(X, transformed_X)
        if embeddings is not None:
            if not self.embeddings:
                raise ValueError("self.embeddings should be True but is not.")

            if isinstance(embeddings, np.ndarray):
                if self.embedding_dimensions["embedding_1"] != embeddings.shape[1]:
                    raise ValueError(
                        f"Expected embedding dimension {self.embedding_dimensions['embedding_1']}, "
                        f"but got {embeddings.shape[1]}"
                    )
                transformed_dict["embedding_1"] = embeddings.astype(np.float32)
            elif isinstance(embeddings, list) and all(
                isinstance(e, np.ndarray) for e in embeddings
            ):
                for idx, e in enumerate(embeddings):
                    key = f"embedding_{idx + 1}"
                    if self.embedding_dimensions[key] != e.shape[1]:
                        raise ValueError(
                            f"Expected embedding dimension {self.embedding_dimensions[key]} for {key}, but got {e.shape[1]}"
                        )
                    transformed_dict[key] = e.astype(np.float32)
        else:
            if self.embeddings is not False:
                raise ValueError(
                    "self.embeddings should be False when embeddings are None."
                )
            self.embeddings = False

        return transformed_dict

    def _split_transformed_output(self, X, transformed_X):
        """Splits the transformed data array into a dictionary where keys correspond to the original column names or
        feature groups and values are the transformed data for those columns.

        This helper method is utilized within `transform` to segregate the transformed data based on the
        specification in the column transformer, assigning each transformed section to its corresponding feature name.

        Parameters
        ----------
            X (DataFrame): The original input data, used for determining shapes and transformations.
            transformed_X (numpy array): The transformed data as a numpy array, outputted by the column transformer.


        Returns
        -------
            dict: A dictionary mapping each transformation's name to its respective numpy array of transformed data.
            The type of each array (int or float) is determined based on the type of transformation applied.
        """
        start = 0
        transformed_dict = {}
        for name, transformer, columns in self.column_transformer.transformers_:  # type: ignore
            if transformer != "drop":
                end = start + transformer.transform(X[[columns[0]]]).shape[1]

                # Determine dtype based on the transformer steps
                transformer_steps = [step[0] for step in transformer.steps]
                if "continuous_ordinal" in transformer_steps:
                    dtype = int  # Use int for ordinal/integer encoding
                else:
                    dtype = float  # Default to float for other encodings

                # Assign transformed data with the correct dtype
                transformed_dict[name] = transformed_X[:, start:end].astype(dtype)
                start = end
        return transformed_dict

    def fit_transform(self, X, y=None, embeddings=None):
        """Fits the preprocessor to the data and then transforms the data using the fitted preprocessing pipelines. This
        is a convenience method that combines `fit` and `transform`.

        Parameters
        ----------
            X (DataFrame or dict): The input dataset to fit the preprocessor on and then transform.
            y (array-like, optional): The target variable. Required if `use_decision_tree_bins` is True.


        Returns
        -------
            dict: A dictionary with the transformed data, where keys are the base feature names and
            values are the transformed features as arrays.
        """
        self.fit(X, y, embeddings)
        self.fitted = True
        return self.transform(X, embeddings)

    def get_feature_info(self, verbose=True):
        """Retrieves information about how features are encoded within the model's preprocessor. This method identifies
        the type of encoding applied to each feature, categorizing them into binned or ordinal encodings and other types
        of encodings (e.g., one-hot encoding after discretization).

        This method should only be called after the preprocessor has been fitted, as it relies on the structure and
        configuration of the `column_transformer` attribute.

        Raises
        ------
            RuntimeError: If the `column_transformer` is not yet fitted, indicating that the preprocessor must be
            fitted before invoking this method.

        Returns
        -------
            tuple of (dict, dict, dict):
                - The first dictionary maps feature names to their respective number of bins or categories if they are
                  processed using discretization or ordinal encoding.
                - The second dictionary includes feature names with other encoding details, such as the dimension of
                  features after encoding transformations (e.g., one-hot encoding dimensions).
                - The third dictionary includes feature information for embeddings if available.
        """
        numerical_feature_info = {}
        categorical_feature_info = {}

        if self.embeddings:
            embedding_feature_info = {}
            for key, dim in self.embedding_dimensions.items():
                embedding_feature_info[key] = {
                    "preprocessing": None,
                    "dimension": dim,
                    "categories": None,
                }
        else:
            embedding_feature_info = {}

        if not self.column_transformer:
            raise RuntimeError("The preprocessor has not been fitted yet.")

        for (
            name,
            transformer_pipeline,
            columns,
        ) in self.column_transformer.transformers_:
            steps = [step[0] for step in transformer_pipeline.steps]

            for feature_name in columns:
                preprocessing_type = " -> ".join(steps)
                dimension = None
                categories = None

                if "discretizer" in steps or any(
                    step in steps
                    for step in [
                        "standardization",
                        "minmax",
                        "quantile",
                        "polynomial",
                        "splines",
                        "box-cox",
                    ]
                ):
                    last_step = transformer_pipeline.steps[-1][1]
                    if hasattr(last_step, "transform"):
                        dummy_input = np.zeros((1, 1)) + 1e-05
                        transformed_feature = last_step.transform(dummy_input)
                        dimension = transformed_feature.shape[1]
                    numerical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": None,
                    }
                    if verbose:
                        print(
                            f"Numerical Feature: {feature_name}, Info: {numerical_feature_info[feature_name]}"
                        )

                elif "continuous_ordinal" in steps:
                    step = transformer_pipeline.named_steps["continuous_ordinal"]
                    categories = len(step.mapping_[columns.index(feature_name)])
                    dimension = 1
                    categorical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": categories,
                    }
                    if verbose:
                        print(
                            f"Categorical Feature (Ordinal): {feature_name}, Info: {categorical_feature_info[feature_name]}"
                        )

                elif "onehot" in steps:
                    step = transformer_pipeline.named_steps["onehot"]
                    if hasattr(step, "categories_"):
                        categories = sum(len(cat) for cat in step.categories_)
                        dimension = categories
                    categorical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": categories,
                    }
                    if verbose:
                        print(
                            f"Categorical Feature (One-Hot): {feature_name}, Info: {categorical_feature_info[feature_name]}"
                        )

                else:
                    last_step = transformer_pipeline.steps[-1][1]
                    if hasattr(last_step, "transform"):
                        dummy_input = np.zeros((1, 1))
                        transformed_feature = last_step.transform(dummy_input)
                        dimension = transformed_feature.shape[1]
                    if "cat" in name:
                        categorical_feature_info[feature_name] = {
                            "preprocessing": preprocessing_type,
                            "dimension": dimension,
                            "categories": None,
                        }
                    else:
                        numerical_feature_info[feature_name] = {
                            "preprocessing": preprocessing_type,
                            "dimension": dimension,
                            "categories": None,
                        }
                    if verbose:
                        print(
                            f"Feature: {feature_name}, Info: {preprocessing_type}, Dimension: {dimension}"
                        )

                if verbose:
                    print("-" * 50)

        if verbose and self.embeddings:
            print("Embeddings:")
            for key, value in embedding_feature_info.items():
                print(f"  Feature: {key}, Dimension: {value['dimension']}")

        return numerical_feature_info, categorical_feature_info, embedding_feature_info
