import numpy as np
from scipy.interpolate import BSpline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class SplineExpansion(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_knots=5,
        degree=3,
        use_decision_tree=True,
        task="regression",
        strategy="quantile",
        spline_implementation="scipy",
    ):
        """
        Initialize the SplineExpansion.

        Parameters:
        - n_knots (int): Number of knots to use for the splines.
        - degree (int): Degree of the splines (e.g., 1 for linear, 3 for cubic).
        - use_decision_tree (bool): If True, use a decision tree to determine knot positions.
        - task (str): Task type, 'regression' or 'classification'.
        - strategy (str): Use 'quantile' or 'uniform' for knot locations calculation when use_decision_tree=False.
        - spline_implementation (str): Use 'scipy' or 'sklearn' for spline implementation.
        """
        self.n_knots = n_knots
        self.degree = degree
        self.use_decision_tree = use_decision_tree
        self.task = task
        self.strategy = strategy
        self.spline_implementation = spline_implementation
        self.knots = None  # To store the knot positions

        if not use_decision_tree and strategy not in ["quantile", "uniform"]:
            raise ValueError(
                "Invalid strategy for knot location calculation. Choose 'quantile' or 'uniform' if decision tree is not used."
            )
        if spline_implementation not in ["scipy", "sklearn"]:
            raise ValueError("Invalid spline implementation. Choose 'scipy' or 'sklearn'.")

    def fit(self, X, y=None):
        """
        Fit the preprocessor by determining the knot positions.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): Input data.
        - y (array-like, shape (n_samples,)): Target values (required if use_decision_tree=True).

        Returns:
        - self: Fitted preprocessor.
        """
        X = np.asarray(X)

        if self.use_decision_tree:
            if y is None:
                raise ValueError("Target variable 'y' must be provided when use_decision_tree=True.")
            y = np.asarray(y)

            self.knots = []
            for i in range(X.shape[1]):
                x_col = X[:, i].reshape(-1, 1)

                # Use DecisionTreeClassifier for classification tasks
                if self.task == "classification":
                    tree = DecisionTreeClassifier(max_leaf_nodes=self.n_knots + 1)
                elif self.task == "regression":
                    tree = DecisionTreeRegressor(max_leaf_nodes=self.n_knots + 1)
                else:
                    raise ValueError("Invalid task type. Choose 'regression' or 'classification'.")

                tree.fit(x_col, y)

                # Extract thresholds from the decision tree
                thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]  # type: ignore
                self.knots.append(np.sort(thresholds))
        else:
            # Compute knots based on uniform spacing or quantile
            self.knots = []
            for i in range(X.shape[1]):
                if self.strategy == "quantile":
                    # Use quantile to determine knot locations
                    quantiles = np.linspace(0, 1, self.n_knots + 2)[1:-1]
                    knots = np.quantile(X[:, i], quantiles)
                    self.knots.append(knots)
                elif self.strategy == "uniform":
                    # Use uniform spacing within the range of the feature
                    knots = np.linspace(np.min(X[:, i]), np.max(X[:, i]), self.n_knots + 2)[1:-1]
                    self.knots.append(knots)

        return self

    def transform(self, X):
        """
        Transform the input data into a higher-dimensional space using splines.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): Input data.

        Returns:
        - X_spline (array-like, shape (n_samples, n_transformed_features)): Transformed data.
        """
        if self.knots is None:
            raise ValueError("Knots have not been initialized. Please fit the preprocessor first.")

        X = np.asarray(X)
        transformed_features = []

        if self.spline_implementation == "scipy":
            for i in range(X.shape[1]):
                x_col = X[:, i]
                knots = self.knots[i]  # type: ignore

                # Extend the knots for boundary conditions
                t = np.concatenate(([knots[0]] * self.degree, knots, [knots[-1]] * self.degree))

                # Create spline basis functions for this feature
                spline_basis = [
                    BSpline.basis_element(t[j : j + self.degree + 2])(x_col) for j in range(len(t) - self.degree - 1)
                ]

                # Stack and append transformed features
                transformed_features.append(np.vstack(spline_basis).T)

                # Concatenate all transformed features
            return np.hstack(transformed_features)
        else:
            if self.use_decision_tree:
                knots = np.vstack(self.knots).T
                transformer = SplineTransformer(
                    n_knots=self.n_knots, degree=self.degree, include_bias=False, knots=knots
                )
            else:
                if self.strategy == "quantile":
                    transformer = SplineTransformer(
                        n_knots=self.n_knots, degree=self.degree, include_bias=False, knots="quantile"
                    )
                elif self.strategy == "uniform":
                    transformer = SplineTransformer(
                        n_knots=self.n_knots, degree=self.degree, include_bias=False, knots="uniform"
                    )
                else:
                    raise ValueError("Invalid strategy for knot location calculation. Choose 'quantile' or 'uniform'.")

            return transformer.fit_transform(X)
