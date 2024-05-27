import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = ['Preprocessor']


class CustomBinner(TransformerMixin):
    def __init__(self, bins):
        # bins can be a scalar (number of bins) or array-like (bin edges)
        self.bins = bins

    def fit(self, X, y=None):
        # Fit doesn't need to do anything as we are directly using provided bins
        return self

    def transform(self, X):
        if isinstance(self.bins, int):
            # Calculate equal width bins based on the range of the data and number of bins
            _, bins = pd.cut(X.squeeze(), bins=self.bins, retbins=True)
        else:
            # Use predefined bins
            bins = self.bins

        # Apply the bins to the data
        binned_data = pd.cut(
            X.squeeze(),
            bins=np.sort(np.unique(bins)),
            labels=False,
            include_lowest=True,
        )
        print(binned_data)
        return np.expand_dims(np.array(binned_data), 1)


class ContinuousOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    This encoder converts categorical features into continuous integer values. Each unique category within a feature
    is assigned a unique integer based on its order of appearance in the dataset. This transformation is useful for
    models that can only handle continuous data.

    Attributes:
        mapping_ (list of dicts): A list where each element is a dictionary mapping original categories to integers
                                  for a single feature.

    Methods:
        fit(X, y=None): Learns the mapping from original categories to integers.
        transform(X): Applies the learned mapping to the data.
        get_feature_names_out(input_features=None): Returns the input features after transformation.
    """

    def fit(self, X, y=None):
        """
        Learns the mapping from original categories to integers for each feature.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to fit.
            y (ignored): Not used, present for API consistency by convention.

        Returns:
            self: Returns the instance itself.
        """
        # Fit should determine the mapping from original categories to sequential integers starting from 0
        self.mapping_ = [
            {category: i for i, category in enumerate(np.unique(col))} for col in X.T
        ]
        return self

    def transform(self, X):
        """
        Transforms the categories in X to their corresponding integer values based on the learned mapping.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to transform.

        Returns:
            X_transformed (ndarray of shape (n_samples, n_features)): The transformed data with integer values.
        """
        # Transform the categories to their mapped integer values
        X_transformed = np.array(
            [
                [self.mapping_[col].get(value, -1)
                 for col, value in enumerate(row)]
                for row in X
            ]
        )
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of the transformed features.

        Parameters:
            input_features (list of str): The names of the input features.

        Returns:
            input_features (array of shape (n_features,)): The names of the output features after transformation.
        """
        if input_features is None:
            raise ValueError("input_features must be specified")
        return input_features


class OneHotFromOrdinal(TransformerMixin, BaseEstimator):
    """
    A transformer that takes ordinal-encoded features and converts them into one-hot encoded format. This is useful
    in scenarios where features have been pre-encoded with ordinal encoding and a one-hot representation is required
    for model training.

    Attributes:
        max_bins_ (ndarray of shape (n_features,)): An array containing the maximum bin index for each feature,
                                                    determining the size of the one-hot encoded array for that feature.

    Methods:
        fit(X, y=None): Learns the maximum bin index for each feature.
        transform(X): Converts ordinal-encoded features into one-hot format.
        get_feature_names_out(input_features=None): Returns the feature names after one-hot encoding.
    """

    def fit(self, X, y=None):
        """
        Learns the maximum bin index for each feature from the data.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to fit, containing ordinal-encoded features.
            y (ignored): Not used, present for API consistency by convention.

        Returns:
            self: Returns the instance itself.
        """
        self.max_bins_ = (
            np.max(X, axis=0).astype(int) + 1
        )  # Find the maximum bin index for each feature
        return self

    def transform(self, X):
        """
        Transforms ordinal-encoded features into one-hot encoded format based on the `max_bins_` learned during fitting.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to transform, containing ordinal-encoded features.

        Returns:
            X_one_hot (ndarray of shape (n_samples, n_output_features)): The one-hot encoded features.
        """
        # Initialize an empty list to hold the one-hot encoded arrays
        one_hot_encoded = []
        for i, max_bins in enumerate(self.max_bins_):
            # Convert each feature to one-hot using its max_bins
            feature_one_hot = np.eye(max_bins)[X[:, i].astype(int)]
            one_hot_encoded.append(feature_one_hot)
        # Concatenate the one-hot encoded features horizontally
        return np.hstack(one_hot_encoded)

    def get_feature_names_out(self, input_features=None):
        """
        Generates feature names for the one-hot encoded features based on the input feature names and the number of bins.

        Parameters:
            input_features (list of str): The names of the input features that were ordinal-encoded.

        Returns:
            feature_names (array of shape (n_output_features,)): The names of the one-hot encoded features.
        """
        feature_names = []
        for i, max_bins in enumerate(self.max_bins_):
            feature_names.extend(
                [f"{input_features[i]}_bin_{j}" for j in range(int(max_bins))]
            )
        return np.array(feature_names)


class Preprocessor:
    """
    A comprehensive preprocessor for structured data, capable of handling both numerical and categorical features.
    It supports various preprocessing strategies for numerical data, including binning, one-hot encoding,
    standardization, and normalization. Categorical features can be transformed using continuous ordinal encoding.
    Additionally, it allows for the use of decision tree-derived bin edges for numerical feature binning.

    The class is designed to work seamlessly with pandas DataFrames, facilitating easy integration into
    machine learning pipelines.

    Parameters
    ----------
        n_bins (int): The number of bins to use for numerical feature binning. This parameter is relevant
                      only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
        numerical_preprocessing (str): The preprocessing strategy for numerical features. Valid options are
                                       'binning', 'one_hot', 'standardization', and 'normalization'.
        use_decision_tree_bins (bool): If True, uses decision tree regression/classification to determine
                                       optimal bin edges for numerical feature binning. This parameter is
                                       relevant only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.

    # Attributes:
    #     column_transformer (ColumnTransformer): A sklearn ColumnTransformer instance that holds the configured
    #                                             preprocessing pipelines for the different feature types.

    # Methods:
    #     fit(X, y=None): Fits the preprocessor to the data, identifying feature types and configuring the
    #                     appropriate transformations.
    #     transform(X): Transforms the data using the fitted preprocessing pipelines.
    #     fit_transform(X, y=None): Fits the preprocessor to the data and then transforms the data.
    #     get_feature_info(): Returns information about the processed features, including the number of bins for
    #                         binned features and the dimensionality of encoded features.
    """

    def __init__(
        self,
        n_bins=200,
        numerical_preprocessing="binning",
        use_decision_tree_bins=False,
        binning_strategy="uniform",
    ):
        self.n_bins = n_bins
        self.numerical_preprocessing = numerical_preprocessing
        self.use_decision_tree_bins = use_decision_tree_bins
        self.column_transformer = None
        self.fitted = False
        self.binning_strategy = binning_strategy

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
                X[col].dtype.kind == "i" and (
                    num_unique_values / total_samples) < 0.05
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
                        self._get_decision_tree_bins(
                            X[[feature]], y, [feature])
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
                                        subsample=200_000 if len(
                                            X) > 200_000 else None,
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
                    numeric_transformer_steps.append(
                        ("scaler", StandardScaler()))

                elif self.numerical_preprocessing == "normalization":
                    numeric_transformer_steps.append(
                        ("normalizer", MinMaxScaler()))

                numeric_transformer = Pipeline(numeric_transformer_steps)

                transformers.append(
                    (f"num_{feature}", numeric_transformer, [feature]))

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
                np.concatenate(
                    ([X[feature].min()], bin_edges, [X[feature].max()]))
            )
        return bins

    def transform(self, X):
        """
        Transforms the dataset using the fitted preprocessing pipelines. This method applies the transformations set up during the fitting process
        to the input data and returns a dictionary with the transformed data.

        Parameters:
            X (DataFrame or dict): The input dataset to be transformed.

        Returns:
            dict: A dictionary where keys are the base feature names and values are the transformed features as arrays.
        """
        if not self.fitted:
            raise NotFittedError(
                "This Preprocessor instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        if isinstance(X, dict):
            X = pd.DataFrame(X)

        # Transform X using the column transformer
        transformed_X = self.column_transformer.transform(
            X
        )  # To understand the shape of the transformed data

        # Initialize the transformed dictionary
        transformed_dict = {}

        # Retrieve output feature names from the column transformer
        output_features = self.column_transformer.get_feature_names_out()

        # Iterate over each output feature name to populate the transformed_dict
        for i, col in enumerate(output_features):
            # Extract the base feature name (before any transformation)
            base_feature = col.split("__")[0]

            # If the base feature name already exists in the dictionary, append the new data
            if base_feature in transformed_dict:
                transformed_dict[base_feature] = np.vstack(
                    [transformed_dict[base_feature], transformed_X[:, i]]
                )
            else:
                # Otherwise, create a new entry in the dictionary
                transformed_dict[base_feature] = transformed_X[:, i]

        # Ensure all arrays in the dictionary are the correct shape
        for key in transformed_dict.keys():
            transformed_dict[key] = (
                transformed_dict[key].reshape(-1, transformed_X.shape[0]).T
            )

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
        Returns detailed information about the processed features, including the number of bins for binned features
        and the dimensionality of encoded features. This method is useful for understanding the transformations applied to each feature.

        Returns:
            tuple: A tuple containing two dictionaries, the first with information about binned or ordinal encoded features and
                   the second with information about other encoded features.
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
                    n_bins = step.n_bins_[0] if hasattr(
                        step, "n_bins_") else None

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
                    n_categories = len(
                        step.mapping_[columns.index(feature_name)])
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
                            f"Feature: {feature_name} ({self.numerical_preprocessing}), Encoded feature dimension: {transformed_feature.shape[1]}"
                        )

                print("-" * 50)

        return binned_or_ordinal_info, other_encoding_info
