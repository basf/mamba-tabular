import numpy as np
from scipy.interpolate import BSpline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_array


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

    @staticmethod
    def knot_identification_using_decision_tree(X, y, task="regression", n_knots=5):
        # Use DecisionTreeClassifier for classification tasks
        knots = []
        if task == "classification":
            tree = DecisionTreeClassifier(max_leaf_nodes=n_knots + 1)
        elif task == "regression":
            tree = DecisionTreeRegressor(max_leaf_nodes=n_knots + 1)
        else:
            raise ValueError("Invalid task type. Choose 'regression' or 'classification'.")
        tree.fit(X, y)
        # Extract thresholds from the decision tree
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]  # type: ignore
        knots.append(np.sort(thresholds))
        return knots

    def fit(self, X, y=None):
        """
        Fit the preprocessor by determining the knot positions.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): Input data.
        - y (array-like, shape (n_samples,)): Target values (required if use_decision_tree=True).

        Returns:
        - self: Fitted preprocessor.
        """
        if self.use_decision_tree and y is None:
            raise ValueError("Target variable 'y' must be provided when use_decision_tree=True.")

        self.knots = []
        self.n_features_in_ = X.shape[1]

        if self.use_decision_tree and self.spline_implementation == "scipy":
            self.knots = self.knot_identification_using_decision_tree(X, y, self.task, self.n_knots)
            self.fitted = True

        elif self.spline_implementation == "scipy" and not self.use_decision_tree:
            if self.strategy == "quantile":
                # Use quantile to determine knot locations
                quantiles = np.linspace(0, 1, self.n_knots + 2)[1:-1]
                knots = np.quantile(X, quantiles)
                self.knots.append(knots)
                self.fitted = True
                # print("Scipy spline implementation using quantile works in fit phase")
            elif self.strategy == "uniform":
                # Use uniform spacing within the range of the feature
                knots = np.linspace(np.min(X), np.max(X), self.n_knots + 2)[1:-1]
                self.knots.append(knots)
                self.fitted = True
                # print("Scipy spline implementation using uniform works in fit phase")

        elif self.use_decision_tree and self.spline_implementation == "sklearn":
            self.knots = self.knot_identification_using_decision_tree(X, y, self.task, self.n_knots)
            knots = np.vstack(self.knots).T
            self.transformer = SplineTransformer(
                n_knots=self.n_knots, degree=self.degree, include_bias=False, knots=knots
            )
            self.transformer.fit(X)
            self.fitted = True

        elif self.spline_implementation == "sklearn" and not self.use_decision_tree:
            if self.strategy == "quantile":
                # print("Using sklearn spline transformer using quantile")
                # print()
                self.transformer = SplineTransformer(
                    n_knots=self.n_knots, degree=self.degree, include_bias=False, knots="quantile"
                )
                self.fitted = True
                self.transformer.fit(X)

            elif self.strategy == "uniform":
                # print("Using sklearn spline transformer using uniform")
                # print()
                self.transformer = SplineTransformer(
                    n_knots=self.n_knots, degree=self.degree, include_bias=False, knots="uniform"
                )
                self.fitted = True
                self.transformer.fit(X)

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

        transformed_features = []

        if self.fitted is False:
            raise ValueError("Model has not been fitted. Please fit the model first.")

        if self.spline_implementation == "scipy":
            # Extend the knots for boundary conditions
            t = np.concatenate(([self.knots[0]] * self.degree, self.knots, [self.knots[-1]] * self.degree))

            # Create spline basis functions for this feature
            spline_basis = [
                BSpline.basis_element(t[j : j + self.degree + 2])(X) for j in range(len(t) - self.degree - 1)
            ]
            # Stack and append transformed features
            transformed_features.append(np.vstack(spline_basis).T)
            # Concatenate all transformed features
            return np.hstack(transformed_features)
        elif self.spline_implementation == "sklearn":
            return self.transformer.transform(X)


def center_identification_using_decision_tree(X, y, task="regression", n_centers=5):
    # Use DecisionTreeClassifier for classification tasks
    centers = []
    if task == "classification":
        tree = DecisionTreeClassifier(max_leaf_nodes=n_centers + 1)
    elif task == "regression":
        tree = DecisionTreeRegressor(max_leaf_nodes=n_centers + 1)
    else:
        raise ValueError("Invalid task type. Choose 'regression' or 'classification'.")
    tree.fit(X, y)
    # Extract thresholds from the decision tree
    thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]  # type: ignore
    centers.append(np.sort(thresholds))
    return centers


class RBFExpansion(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_centers=10, gamma: float = 1.0, use_decision_tree=True, task: str = "regression", strategy="uniform"
    ):
        """
        Radial Basis Function Expansion.

        Parameters:
        - n_centers: Number of RBF centers.
        - gamma: Width of the RBF kernel.
        - use_decision_tree: If True, use a decision tree to determine RBF centers.
        - task: Task type, 'regression' or 'classification'.
        - strategy: If 'uniform', centers are uniformly spaced. If 'quantile', centers are
                    determined by data quantile.
        """
        self.n_centers = n_centers
        self.gamma = gamma
        self.use_decision_tree = use_decision_tree
        self.strategy = strategy
        self.task = task

        if self.strategy not in ["uniform", "quantile"]:
            raise ValueError("Invalid strategy. Choose 'uniform' or 'quantile'.")

    def fit(self, X, y=None):
        X = check_array(X)

        if self.use_decision_tree and y is None:
            raise ValueError("Target variable 'y' must be provided when use_decision_tree=True.")

        if self.use_decision_tree:
            self.centers_ = center_identification_using_decision_tree(X, y, self.task, self.n_centers)
            self.centers_ = np.vstack(self.centers_)
        else:
            # Compute centers
            if self.strategy == "quantile":
                self.centers_ = np.percentile(X, np.linspace(0, 100, self.n_centers), axis=0)
            elif self.strategy == "uniform":
                self.centers_ = np.linspace(X.min(axis=0), X.max(axis=0), self.n_centers)

        # Compute gamma if not provided
        # if self.gamma is None:
        #     dists = pairwise_distances(self.centers_)
        #     self.gamma = 1 / (2 * np.mean(dists[dists > 0]) ** 2)  # Mean pairwise distance
        return self

    def transform(self, X):
        X = check_array(X)
        transformed = []
        self.centers_ = np.array(self.centers_)
        for center in self.centers_.T:
            rbf_features = np.exp(-self.gamma * (X - center) ** 2)  # type: ignore
            transformed.append(rbf_features)
        return np.hstack(transformed)


class SigmoidExpansion(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_centers=10, scale: float = 1.0, use_decision_tree=True, task: str = "regression", strategy="uniform"
    ):
        """
        Sigmoid Basis Expansion.

        Parameters:
        - n_centers: Number of sigmoid centers.
        - scale: Scale parameter for sigmoid function.
        - use_decision_tree: If True, use a decision tree to determine sigmoid centers.
        - task: Task type, 'regression' or 'classification'.
        - strategy: If 'uniform', centers are uniformly spaced. If 'quantile', centers are
                    determined by data quantile.
        """
        self.n_centers = n_centers
        self.scale = scale
        self.use_decision_tree = use_decision_tree
        self.strategy = strategy
        self.task = task

    def fit(self, X, y=None):
        X = check_array(X)

        if self.use_decision_tree and y is None:
            raise ValueError("Target variable 'y' must be provided when use_decision_tree=True.")

        if self.use_decision_tree:
            self.centers_ = center_identification_using_decision_tree(X, y, self.task, self.n_centers)
            self.centers_ = np.vstack(self.centers_)
        else:
            # Compute centers
            if self.strategy == "quantile":
                self.centers_ = np.percentile(X, np.linspace(0, 100, self.n_centers), axis=0)
            elif self.strategy == "uniform":
                self.centers_ = np.linspace(X.min(axis=0), X.max(axis=0), self.n_centers)

        # Compute gamma if not provided
        # if self.gamma is None:
        #     dists = pairwise_distances(self.centers_)
        #     self.gamma = 1 / (2 * np.mean(dists[dists > 0]) ** 2)  # Mean pairwise distance
        return self

    def transform(self, X):
        X = check_array(X)
        transformed = []

        self.centers_ = np.array(self.centers_)
        for center in self.centers_.T:
            sigmoid_features = 1 / (1 + np.exp(-(X - center) / self.scale))
            transformed.append(sigmoid_features)
        return np.hstack(transformed)
