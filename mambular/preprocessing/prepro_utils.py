import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomBinner(TransformerMixin, BaseEstimator):
    def __init__(self, bins):
        # bins can be a scalar (number of bins) or array-like (bin edges)
        self.bins = bins

    def fit(self, X, y=None):
        # Fit doesn't need to do anything as we are directly using provided bins
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        if isinstance(self.bins, int):
            # Calculate equal width bins based on the range of the data and number of bins
            _, bins = pd.cut(X.squeeze(), bins=self.bins, retbins=True)
        else:
            # Use predefined bins
            bins = self.bins

        # Apply the bins to the data
        binned_data = pd.cut(  # type: ignore
            X.squeeze(),
            bins=np.sort(np.unique(bins)),  # type: ignore
            labels=False,
            include_lowest=True,
        )
        return np.expand_dims(np.array(binned_data), 1)

    def get_feature_names_out(self, input_features=None):
        """Returns the names of the transformed features.

        Parameters:
            input_features (list of str): The names of the input features.

        Returns:
            input_features (array of shape (n_features,)): The names of the output features after transformation.
        """
        if input_features is None:
            raise ValueError("input_features must be specified")
        return input_features


class ContinuousOrdinalEncoder(BaseEstimator, TransformerMixin):
    """This encoder converts categorical features into continuous integer values. Each unique category within a feature
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
        """Learns the mapping from original categories to integers for each feature.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to fit.
            y (ignored): Not used, present for API consistency by convention.

        Returns:
            self: Returns the instance itself.
        """
        # Fit should determine the mapping from original categories to sequential integers starting from 0
        self.mapping_ = [
            {category: i + 1 for i, category in enumerate(np.unique(col))}
            for col in X.T
        ]
        for mapping in self.mapping_:
            mapping[None] = 0  # Assign 0 to unknown values
        return self

    def transform(self, X):
        """Transforms the categories in X to their corresponding integer values based on the learned mapping.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to transform.

        Returns:
            X_transformed (ndarray of shape (n_samples, n_features)): The transformed data with integer values.
        """
        # Transform the categories to their mapped integer values
        X_transformed = np.array(
            [
                [self.mapping_[col].get(value, 0) for col, value in enumerate(row)]
                for row in X
            ]
        )
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Returns the names of the transformed features.

        Parameters:
            input_features (list of str): The names of the input features.

        Returns:
            input_features (array of shape (n_features,)): The names of the output features after transformation.
        """
        if input_features is None:
            raise ValueError("input_features must be specified")
        return input_features


class OneHotFromOrdinal(TransformerMixin, BaseEstimator):
    """A transformer that takes ordinal-encoded features and converts them into one-hot encoded format. This is useful
    in scenarios where features have been pre-encoded with ordinal encoding and a one-hot representation is required for
    model training.

    Attributes:
        max_bins_ (ndarray of shape (n_features,)): An array containing the maximum bin index for each feature,
                                                    determining the size of the one-hot encoded array for that feature.

    Methods:
        fit(X, y=None): Learns the maximum bin index for each feature.
        transform(X): Converts ordinal-encoded features into one-hot format.
        get_feature_names_out(input_features=None): Returns the feature names after one-hot encoding.
    """

    def fit(self, X, y=None):
        """Learns the maximum bin index for each feature from the data.

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
        """Transforms ordinal-encoded features into one-hot encoded format based on the `max_bins_` learned during
        fitting.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to transform,
            containing ordinal-encoded features.

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
        """Generates feature names for the one-hot encoded features based on the input feature names and the number of
        bins.

        Parameters:
            input_features (list of str): The names of the input features that were ordinal-encoded.

        Returns:
            feature_names (array of shape (n_output_features,)): The names of the one-hot encoded features.
        """
        feature_names = []
        for i, max_bins in enumerate(self.max_bins_):
            feature_names.extend([f"{input_features[i]}_bin_{j}" for j in range(int(max_bins))])  # type: ignore
        return np.array(feature_names)


class NoTransformer(TransformerMixin, BaseEstimator):
    """A transformer that does not preprocess the data but retains compatibility with the sklearn pipeline API. It
    simply returns the input data as is.

    Methods:
        fit(X, y=None): Fits the transformer to the data (no operation).
        transform(X): Returns the input data unprocessed.
        get_feature_names_out(input_features=None): Returns the original feature names.
    """

    def fit(self, X, y=None):
        """Fits the transformer to the data. No operation is performed.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to fit.
            y (ignored): Not used, present for API consistency by convention.

        Returns:
            self: Returns the instance itself.
        """
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        """Returns the input data unprocessed.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to transform.

        Returns:
            X (array-like): The same input data, unmodified.
        """
        return X

    def get_feature_names_out(self, input_features=None):
        """Returns the original feature names.

        Parameters:
            input_features (list of str or None): The names of the input features.

        Returns:
            feature_names (array of shape (n_features,)): The original feature names.
        """
        if input_features is None:
            raise ValueError(
                "input_features must be provided to generate feature names."
            )
        return np.array(input_features)


class ToFloatTransformer(TransformerMixin, BaseEstimator):
    """A transformer that converts input data to float type."""

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        return X.astype(float)


class LanguageEmbeddingTransformer(TransformerMixin, BaseEstimator):
    """A transformer that encodes categorical text features into embeddings using a pre-trained language model."""

    def __init__(self, model_name="paraphrase-MiniLM-L3-v2", model=None):
        """
        Initializes the transformer with a language embedding model.

        Parameters:
        - model_name (str): The name of the SentenceTransformer model to use (if model is None).
        - model (object, optional): A preloaded SentenceTransformer model instance.
        """
        self.model_name = model_name
        self.model = model  # Allow user to pass a preloaded model

        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed. Install it via `pip install sentence-transformers` or provide a preloaded model."
                ) from e

    def fit(self, X, y=None):
        """Fit method (not required for a transformer but included for compatibility)."""
        self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1
        return self

    def transform(self, X):
        """
        Transforms input categorical text features into numerical embeddings.

        Parameters:
        - X: A 1D or 2D array-like of categorical text features.

        Returns:
        - A 2D numpy array with embeddings for each text input.
        """
        if isinstance(X, np.ndarray):
            X = (
                X.flatten().astype(str).tolist()
            )  # Convert to a list of strings if passed as an array
        elif isinstance(X, list):
            X = [str(x) for x in X]  # Ensure everything is a string

        if self.model is None:
            raise ValueError(
                "Model is not initialized. Ensure that the model is properly loaded."
            )
        embeddings = self.model.encode(
            X, convert_to_numpy=True
        )  # Get sentence embeddings
        return embeddings
