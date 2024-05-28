import pandas as pd
import numpy as np
from .prepro_utils import OneHotFromOrdinal, CustomBinner, ContinuousOrdinalEncoder
from sklearn.preprocessing import (
    StandardScaler,
    KBinsDiscretizer,
    MinMaxScaler,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .ple_encoding import PLE


class Preprocessor:
    """
    A comprehensive preprocessor for structured data, capable of handling both numerical and categorical features.
    It supports various preprocessing strategies for numerical data, including binning, one-hot encoding,
    standardization, and normalization. Categorical features can be transformed using continuous ordinal encoding.
    Additionally, it allows for the use of decision tree-derived bin edges for numerical feature binning.

    The class is designed to work seamlessly with pandas DataFrames, facilitating easy integration into
    machine learning pipelines.

    Parameters:
        n_bins (int): The number of bins to use for numerical feature binning. This parameter is relevant
                      only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
        numerical_preprocessing (str): The preprocessing strategy for numerical features. Valid options are
                                       'binning', 'one_hot', 'standardization', and 'normalization'.
        use_decision_tree_bins (bool): If True, uses decision tree regression/classification to determine
                                       optimal bin edges for numerical feature binning. This parameter is
                                       relevant only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
        binning_strategy (str): Defines the strategy for binning numerical features. Options include 'uniform',
                                'quantile', or other sklearn-compatible strategies.
        task (str): Indicates the type of machine learning task ('regression' or 'classification'). This can
                    influence certain preprocessing behaviors, especially when using decision tree-based binning.

    Attributes:
        column_transformer (ColumnTransformer): An instance of sklearn's ColumnTransformer that holds the
                                                configured preprocessing pipelines for different feature types.
        fitted (bool): Indicates whether the preprocessor has been fitted to the data.

    Methods:
        fit(X, y=None): Fits the preprocessor to the data, identifying feature types and configuring the
                        appropriate transformations.
        transform(X): Transforms the data using the fitted preprocessing pipelines.
        fit_transform(X, y=None): Fits the preprocessor to the data and then transforms the data.
        get_feature_info(): Returns information about the processed features, including the number of bins for
                            binned features and the dimensionality of encoded features.
    """

    def __init__(
        self,
        n_bins=200,
        numerical_preprocessing="binning",
        use_decision_tree_bins=False,
        binning_strategy="uniform",
        task="regression",
    ):
        self.n_bins = n_bins
        self.numerical_preprocessing = numerical_preprocessing.lower()
        if self.numerical_preprocessing not in [
            "ple",
            "binning",
            "one_hot",
            "standardization",
            "normalization",
        ]:
            raise ValueError(
                "Invalid numerical_preprocessing value. Supported values are 'ple', 'binning', 'one_hot', 'standardization', and 'normalization'."
            )
        self.use_decision_tree_bins = use_decision_tree_bins
        self.column_transformer = None
        self.fitted = False
        self.binning_strategy = binning_strategy
        self.task = task

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _detect_column_types(self, X):
        """
        Identifies and separates the features in the dataset into numerical and categorical types based on the data type
        and the proportion of unique values.

        Parameters:
            X (DataFrame or dict): The input dataset, where the features are columns in a DataFrame or keys in a dict.

        Returns:
            tuple: A tuple containing two lists, the first with the names of numerical features and the second with the names of categorical features.
        """
        categorical_features = []
        numerical_features = []

        if isinstance(X, dict):
            X = pd.DataFrame(X)

        for col in X.columns:
            num_unique_values = X[col].nunique()
            total_samples = len(X[col])
            if X[col].dtype.kind not in "iufc" or (
                X[col].dtype.kind == "i" and (num_unique_values / total_samples) < 0.05
            ):
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        return numerical_features, categorical_features

    def fit(self, X, y=None):
        """
        Fits the preprocessor to the data by identifying feature types and configuring the appropriate transformations for each feature.
        It sets up a column transformer with a pipeline of transformations for numerical and categorical features based on the specified preprocessing strategy.

        Parameters:
            X (DataFrame or dict): The input dataset to fit the preprocessor on.
            y (array-like, optional): The target variable. Required if `use_decision_tree_bins` is True for determining optimal bin edges using decision trees.

        Returns:
            self: The fitted Preprocessor instance.
        """
        if isinstance(X, dict):
            X = pd.DataFrame(X)

        numerical_features, categorical_features = self._detect_column_types(X)
        transformers = []

        if numerical_features:
            for feature in numerical_features:
                numeric_transformer_steps = [
                    ("imputer", SimpleImputer(strategy="mean"))
                ]

                if self.numerical_preprocessing in ["binning", "one_hot"]:
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
                                        n_bins=bins
                                        if isinstance(bins, int)
                                        else len(bins) - 1,
                                        encode="ordinal",
                                        strategy=self.binning_strategy,
                                        subsample=200_000 if len(X) > 200_000 else None,
                                    ),
                                ),
                            ]
                        )
                    else:
                        numeric_transformer_steps.extend(
                            [
                                (
                                    "discretizer",
                                    CustomBinner(bins=bins),
                                ),
                            ]
                        )

                    if self.numerical_preprocessing == "one_hot":
                        numeric_transformer_steps.extend(
                            [
                                ("onehot_from_ordinal", OneHotFromOrdinal()),
                            ]
                        )

                elif self.numerical_preprocessing == "standardization":
                    numeric_transformer_steps.append(("scaler", StandardScaler()))

                elif self.numerical_preprocessing == "normalization":
                    numeric_transformer_steps.append(("normalizer", MinMaxScaler()))

                elif self.numerical_preprocessing == "ple":
                    numeric_transformer_steps.append(("normalizer", MinMaxScaler()))
                    numeric_transformer_steps.append(
                        ("ple", PLE(n_bins=self.n_bins, task=self.task))
                    )

                numeric_transformer = Pipeline(numeric_transformer_steps)

                transformers.append((f"num_{feature}", numeric_transformer, [feature]))

        if categorical_features:
            for feature in categorical_features:
                # Create a pipeline for each categorical feature
                categorical_transformer = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("continuous_ordinal", ContinuousOrdinalEncoder()),
                    ]
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
        """
        Uses decision tree models to determine optimal bin edges for numerical feature binning. This method is used when `use_decision_tree_bins` is True.

        Parameters:
            X (DataFrame): The input dataset containing only the numerical features for which the bin edges are to be determined.
            y (array-like): The target variable for training the decision tree models.
            numerical_features (list of str): The names of the numerical features for which the bin edges are to be determined.

        Returns:
            list: A list of arrays, where each array contains the bin edges determined by the decision tree for a numerical feature.
        """
        bins = []
        for feature in numerical_features:
            tree_model = (
                DecisionTreeClassifier(max_depth=3)
                if y.dtype.kind in "bi"
                else DecisionTreeRegressor(max_depth=3)
            )
            tree_model.fit(X[[feature]], y)
            thresholds = tree_model.tree_.threshold[tree_model.tree_.feature != -2]
            bin_edges = np.sort(np.unique(thresholds))

            bins.append(
                np.concatenate(([X[feature].min()], bin_edges, [X[feature].max()]))
            )
        return bins

    def transform(self, X):
        """
        Transforms the input data using the preconfigured column transformer and converts the output into a dictionary
        format with keys corresponding to transformed feature names and values as arrays of transformed data.

        This method converts the sparse or dense matrix returned by the column transformer into a more accessible
        dictionary format, where each key-value pair represents a feature and its transformed data.

        Parameters:
            X (DataFrame): The input data to be transformed.

        Returns:
            dict: A dictionary where keys are the names of the features (as per the transformations defined in the
            column transformer) and the values are numpy arrays of the transformed data.
        """
        transformed_X = self.column_transformer.transform(X)

        # Now let's convert this into a dictionary of arrays, one per column
        transformed_dict = self._split_transformed_output(X, transformed_X)
        return transformed_dict

    def _split_transformed_output(self, X, transformed_X):
        """
        Splits the transformed data array into a dictionary where keys correspond to the original column names or
        feature groups and values are the transformed data for those columns.

        This helper method is utilized within `transform` to segregate the transformed data based on the
        specification in the column transformer, assigning each transformed section to its corresponding feature name.

        Parameters:
            X (DataFrame): The original input data, used for determining shapes and transformations.
            transformed_X (numpy array): The transformed data as a numpy array, outputted by the column transformer.

        Returns:
            dict: A dictionary mapping each transformation's name to its respective numpy array of transformed data.
            The type of each array (int or float) is determined based on the type of transformation applied.
        """
        start = 0
        transformed_dict = {}
        for (
            name,
            transformer,
            columns,
        ) in self.column_transformer.transformers_:  # skip 'remainder'
            if transformer != "drop":
                end = start + transformer.transform(X[[columns[0]]]).shape[1]
                dtype = int if "cat" in name else float
                transformed_dict[name] = transformed_X[:, start:end].astype(dtype)
                start = end

        return transformed_dict

    def fit_transform(self, X, y=None):
        """
        Fits the preprocessor to the data and then transforms the data using the fitted preprocessing pipelines. This is a convenience method that combines `fit` and `transform`.

        Parameters:
            X (DataFrame or dict): The input dataset to fit the preprocessor on and then transform.
            y (array-like, optional): The target variable. Required if `use_decision_tree_bins` is True.

        Returns:
            dict: A dictionary with the transformed data, where keys are the base feature names and values are the transformed features as arrays.
        """
        self.fit(X, y)
        self.fitted = True
        return self.transform(X)

    def get_feature_info(self):
        """
        Retrieves information about how features are encoded within the model's preprocessor.
        This method identifies the type of encoding applied to each feature, categorizing them into binned or ordinal
        encodings and other types of encodings (e.g., one-hot encoding after discretization).

        This method should only be called after the preprocessor has been fitted, as it relies on the structure and
        configuration of the `column_transformer` attribute.

        Raises:
            RuntimeError: If the `column_transformer` is not yet fitted, indicating that the preprocessor must be
            fitted before invoking this method.

        Returns:
            tuple of (dict, dict):
                - The first dictionary maps feature names to their respective number of bins or categories if they are
                  processed using discretization or ordinal encoding.
                - The second dictionary includes feature names with other encoding details, such as the dimension of
                  features after encoding transformations (e.g., one-hot encoding dimensions).
        """
        binned_or_ordinal_info = {}
        other_encoding_info = {}

        if not self.column_transformer:
            raise RuntimeError("The preprocessor has not been fitted yet.")

        for (
            name,
            transformer_pipeline,
            columns,
        ) in self.column_transformer.transformers_:
            steps = [step[0] for step in transformer_pipeline.steps]

            for feature_name in columns:
                # Handle features processed with discretization
                if "discretizer" in steps:
                    step = transformer_pipeline.named_steps["discretizer"]
                    n_bins = step.n_bins_[0] if hasattr(step, "n_bins_") else None

                    # Check if discretization is followed by one-hot encoding
                    if "onehot_from_ordinal" in steps:
                        # Classify as other encoding due to the expanded feature dimensions from one-hot encoding
                        other_encoding_info[
                            feature_name
                        ] = n_bins  # Number of bins before one-hot encoding
                        print(
                            f"Numerical Feature (Discretized & One-Hot Encoded): {feature_name}, Number of bins before one-hot encoding: {n_bins}"
                        )
                    else:
                        # Only discretization without subsequent one-hot encoding
                        binned_or_ordinal_info[feature_name] = n_bins
                        print(
                            f"Numerical Feature (Binned): {feature_name}, Number of bins: {n_bins}"
                        )

                # Handle features processed with continuous ordinal encoding
                elif "continuous_ordinal" in steps:
                    step = transformer_pipeline.named_steps["continuous_ordinal"]
                    n_categories = len(step.mapping_[columns.index(feature_name)])
                    binned_or_ordinal_info[feature_name] = n_categories
                    print(
                        f"Categorical Feature (Ordinal Encoded): {feature_name}, Number of unique categories: {n_categories}"
                    )

                # Handle other numerical feature encodings
                else:
                    last_step = transformer_pipeline.steps[-1][1]
                    if hasattr(last_step, "transform"):
                        transformed_feature = last_step.transform(
                            np.zeros((1, len(columns)))
                        )
                        other_encoding_info[feature_name] = transformed_feature.shape[1]
                        print(
                            f"Feature: {feature_name} (Other Encoding), Encoded feature dimension: {transformed_feature.shape[1]}"
                        )

                print("-" * 50)

        return binned_or_ordinal_info, other_encoding_info
