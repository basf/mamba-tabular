import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from mambular.models import MambularRegressor  # Ensure correct import path


class TestMambularRegressor(unittest.TestCase):
    def setUp(self):
        # Patching external dependencies
        self.patcher_pl_trainer = patch("lightning.Trainer")
        self.mock_pl_trainer = self.patcher_pl_trainer.start()

        self.patcher_base_model = patch(
            "mambular.base_models.regressor.BaseMambularRegressor"
        )
        self.mock_base_model = self.patcher_base_model.start()

        self.regressor = MambularRegressor(d_model=128, dropout=0.1)

        # Sample data
        self.X = pd.DataFrame(np.random.randn(100, 10))
        self.y = np.random.rand(100)

        self.regressor.cat_feature_info = {}
        self.regressor.num_feature_info = {}

    def tearDown(self):
        self.patcher_pl_trainer.stop()
        self.patcher_base_model.stop()

    def test_initialization(self):
        # This assumes MambularConfig is properly imported and used in the MambularRegressor class
        from mambular.utils.configs import DefaultMambularConfig

        self.assertIsInstance(self.regressor.config, DefaultMambularConfig)
        self.assertEqual(self.regressor.config.d_model, 128)
        self.assertEqual(self.regressor.config.dropout, 0.1)

    def test_split_data(self):
        """Test the data splitting functionality."""
        X_train, X_val, y_train, y_val = self.regressor.split_data(
            self.X, self.y, val_size=0.2, random_state=42
        )
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_val), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_val), 20)

    def test_fit(self):
        """Test the training setup and call."""
        # Mock the necessary parts to simulate training
        self.regressor.preprocess_data = MagicMock()
        self.regressor.model = self.mock_base_model

        self.regressor.fit(self.X, self.y)

        # Ensure that the fit method of the trainer is called
        self.mock_pl_trainer.return_value.fit.assert_called_once()

    def test_predict(self):
        # Create mock return objects that mimic tensor behavior
        mock_prediction = MagicMock()
        mock_prediction.cpu.return_value = MagicMock()
        mock_prediction.cpu.return_value.numpy.return_value = np.array([0.5] * 100)

        # Mock the model and its method calls
        self.regressor.model = MagicMock()
        self.regressor.model.eval.return_value = None
        self.regressor.model.return_value = mock_prediction

        # Mock preprocess_test_data to return dummy tensor data
        self.regressor.preprocess_test_data = MagicMock(return_value=([], []))

        predictions = self.regressor.predict(self.X)

        # Assert that predictions return as expected
        np.testing.assert_array_equal(predictions, np.array([0.5] * 100))

    def test_evaluate(self):
        # Mock the predict method to simulate regressor output
        mock_predictions = np.random.rand(100)
        self.regressor.predict = MagicMock(return_value=mock_predictions)

        # Define metrics to test
        metrics = {"Mean Squared Error": mean_squared_error, "R2 Score": r2_score}

        # Call evaluate with the defined metrics
        result = self.regressor.evaluate(self.X, self.y, metrics=metrics)

        # Compute expected metrics directly
        expected_mse = mean_squared_error(self.y, mock_predictions)
        expected_r2 = r2_score(self.y, mock_predictions)

        # Check the results of evaluate
        self.assertAlmostEqual(result["Mean Squared Error"], expected_mse)
        self.assertAlmostEqual(result["R2 Score"], expected_r2)

        # Ensure predict was called correctly
        self.regressor.predict.assert_called_once_with(self.X)


if __name__ == "__main__":
    unittest.main()
