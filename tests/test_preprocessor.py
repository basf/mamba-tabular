import unittest
import pandas as pd
import numpy as np
import sys
from mambular.utils.preprocessor import Preprocessor
from sklearn.exceptions import NotFittedError


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame(
            {
                "numerical": np.random.randn(500),
                "categorical": np.random.choice(["A", "B", "C"], size=500),
                "mixed": np.random.choice([1, "A", "B"], size=500),
            }
        )
        self.target = np.random.randn(500)

    def test_initialization(self):
        """Test initialization of the Preprocessor with default parameters."""
        pp = Preprocessor(n_bins=20, numerical_preprocessing="binning")
        self.assertEqual(pp.n_bins, 20)
        self.assertEqual(pp.numerical_preprocessing, "binning")
        self.assertFalse(pp.use_decision_tree_bins)

    def test_fit(self):
        """Test the fitting process of the preprocessor."""
        pp = Preprocessor(numerical_preprocessing="binning", n_bins=20)
        pp.fit(self.data, self.target)
        self.assertIsNotNone(pp.column_transformer)

    def test_transform_not_fitted(self):
        """Test that transform raises an error if called before fitting."""
        pp = Preprocessor()
        with self.assertRaises(NotFittedError):
            pp.transform(self.data)

    def test_fit_transform(self):
        """Test fitting and transforming the data."""
        pp = Preprocessor(numerical_preprocessing="standardization")
        transformed_data = pp.fit_transform(self.data, self.target)
        self.assertIsInstance(transformed_data, dict)
        self.assertTrue("num_numerical" in transformed_data)
        self.assertTrue("cat_categorical" in transformed_data)

    def test_ple(self):
        """Test fitting and transforming the data."""
        pp = Preprocessor(numerical_preprocessing="ple", n_bins=20)
        transformed_data = pp.fit_transform(self.data, self.target)
        self.assertIsInstance(transformed_data, dict)
        self.assertTrue("num_numerical" in transformed_data)
        self.assertTrue("cat_categorical" in transformed_data)

    def test_transform_with_missing_values(self):
        """Ensure the preprocessor can handle missing values."""
        data_with_missing = self.data.copy()
        data_with_missing.loc[0, "numerical"] = np.nan
        data_with_missing.loc[1, "categorical"] = np.nan
        pp = Preprocessor(numerical_preprocessing="normalization")
        transformed_data = pp.fit_transform(data_with_missing, self.target)
        self.assertNotIn(np.nan, transformed_data["num_numerical"])
        self.assertNotIn(np.nan, transformed_data["cat_categorical"])

    def test_decision_tree_bins(self):
        """Test the usage of decision tree for binning."""
        pp = Preprocessor(
            use_decision_tree_bins=True, numerical_preprocessing="binning", n_bins=5
        )
        pp.fit(self.data, self.target)
        # Checking if the preprocessor setup decision tree bins properly
        self.assertTrue(
            all(
                isinstance(x, np.ndarray)
                for x in pp._get_decision_tree_bins(
                    self.data[["numerical"]], self.target, ["numerical"]
                )
            )
        )


# Running the tests
if __name__ == "__main__":
    unittest.main()
