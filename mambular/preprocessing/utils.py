import pandas as pd
import numpy as np
import warnings


def check_inputs(
    X,
    y=None,
    numerical_columns=None,
    categorical_columns=None,
    task_type=None,
    min_samples=5,
):
    """
    Perform thorough validation on input features and target.

    Parameters
    ----------
    X : pd.DataFrame or dict
        Input features.
    y : array-like, optional
        Target values.
    numerical_columns : list of str
        Columns expected to be numerical.
    categorical_columns : list of str
        Columns expected to be categorical.
    task_type : str, optional
        One of {"regression", "binary", "multiclass"}. If specified, target checks will apply accordingly.
    min_samples : int, optional
        Minimum number of distinct values required in any feature or target.

    Raises
    ------
    ValueError
        If any feature or target fails validation checks.
    """
    if isinstance(X, dict):
        X = pd.DataFrame(X)

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a DataFrame or a dict convertible to DataFrame.")

    if X.empty:
        raise ValueError("X must not be empty.")

    if numerical_columns is None:
        numerical_columns = []
    if categorical_columns is None:
        categorical_columns = []

    all_cols = set(numerical_columns) | set(categorical_columns)
    missing_cols = all_cols - set(X.columns)
    if missing_cols:
        raise ValueError(
            f"The following specified columns are missing in X: {missing_cols}"
        )

    # Check numerical features
    for col in numerical_columns:
        series = X[col]
        if series.nunique(dropna=False) < min_samples:
            raise ValueError(
                f"Numerical feature '{col}' has less than {min_samples} unique values."
            )
        if not np.issubdtype(series.dtype, np.number):
            raise TypeError(f"Numerical feature '{col}' must be numeric.")
        if not np.all(np.isfinite(series.dropna())):
            raise ValueError(
                f"Numerical feature '{col}' contains non-finite values (inf or NaN)."
            )

    # Check categorical features
    for col in categorical_columns:
        series = X[col]
        if series.nunique(dropna=False) < 2:
            raise ValueError(
                f"Categorical feature '{col}' has less only a single value ."
            )
        if pd.api.types.is_numeric_dtype(
            series
        ) and not pd.api.types.is_categorical_dtype(series):
            # allow numerical dtypes only if user intends to encode them
            pass  # optionally warn or convert
        if series.isnull().all():
            raise ValueError(f"Categorical feature '{col}' contains only NaNs.")

    # Check y
    if y is not None:
        y = np.array(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array or Series.")

        if len(y) != len(X):
            raise ValueError("X and y must have the same number of samples.")

        unique_targets = np.unique(y[~pd.isnull(y)])
        n_classes = len(unique_targets)

        if task_type == "regression":
            if not np.issubdtype(y.dtype, np.number):
                raise TypeError("For regression, target y must be numeric.")
            if not np.all(np.isfinite(y)):
                raise ValueError("Target y contains non-finite values.")

            if n_classes <= 10:
                warnings.warn(
                    f"Target y has only {n_classes} unique values. "
                    "Consider if this should be a classification problem instead of regression.",
                    UserWarning,
                )

        elif task_type == "classification":
            if n_classes < 2:
                raise ValueError("Classification tasks requires more than 1 class.")
