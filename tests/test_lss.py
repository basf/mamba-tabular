import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from properscoring import (
    crps_gaussian,
)  # Assuming this is the source of the CRPS function
from sklearn.metrics import mean_poisson_deviance, mean_squared_error

from mambular.models import MambularLSS  # Update the import path


class TestMambularLSS(unittest.TestCase):
    def setUp(self):
        # Patch PyTorch Lightning's Trainer and any other external dependencies
        self.patcher_trainer = patch("lightning.Trainer")
        self.mock_trainer = self.patcher_trainer.start()

        self.patcher_base_model = patch(
            "mambular.base_models.distributional.BaseMambularLSS"
        )
        self.mock_base_model = self.patcher_base_model.start()

        # Initialize MambularLSS with example parameters
        self.model = MambularLSS(d_model=128, dropout=0.1, n_layers=4)

        # Sample data
        self.X = pd.DataFrame(np.random.randn(100, 10))
        self.y = np.random.rand(100)

        self.model.cat_feature_info = {}
        self.model.num_feature_info = {}

        self.X_test = pd.DataFrame(np.random.randn(100, 10))
        self.y_test = np.random.rand(100) ** 2

    def tearDown(self):
        self.patcher_trainer.stop()
        self.patcher_base_model.stop()

    def test_initialization(self):
        from mambular.utils.configs import DefaultMambularConfig

        self.assertIsInstance(self.model.config, DefaultMambularConfig)
        self.assertEqual(self.model.config.d_model, 128)
        self.assertEqual(self.model.config.dropout, 0.1)
        self.assertEqual(self.model.config.n_layers, 4)

    def test_split_data(self):
        X_train, X_val, y_train, y_val = self.model.split_data(
            self.X, self.y, val_size=0.2, random_state=42
        )
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_val), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_val), 20)

    def test_fit(self):
        # Mock preprocessing and model setup to focus on testing training logic
        self.model.preprocess_data = MagicMock()
        self.model.model = self.mock_base_model

        self.model.fit(self.X, self.y, family="normal")

        # Ensure the fit method of the trainer is called
        self.mock_trainer.return_value.fit.assert_called_once()

    def test_predict(self):
        # Create a mock tensor as model output
        mock_prediction = torch.rand(100)
        self.model.model = MagicMock()
        self.model.model.return_value = mock_prediction
        self.model.preprocess_test_data = MagicMock(return_value=([], []))

        predictions = self.model.predict(self.X)

        # Convert tensor to numpy and check equality
        np.testing.assert_array_equal(predictions, mock_prediction.numpy())

    def test_normal_metrics(self):
        # Mock predictions for the normal distribution: [mean, variance]
        mock_predictions = np.column_stack(
            (np.random.normal(size=100), np.abs(np.random.normal(size=100)))
        )
        self.model.predict = MagicMock(return_value=mock_predictions)

        # Define custom metrics or use a function that fetches appropriate metrics
        self.model.get_default_metrics = MagicMock(
            return_value={
                "MSE": lambda y, pred: mean_squared_error(y, pred[:, 0]),
                "CRPS": lambda y, pred: np.mean(
                    [
                        crps_gaussian(y[i], mu=pred[i, 0], sig=np.sqrt(pred[i, 1]))
                        for i in range(len(y))
                    ]
                ),
            }
        )

        results = self.model.evaluate(
            self.X_test, self.y_test, distribution_family="normal"
        )

        # Validate the MSE
        expected_mse = mean_squared_error(self.y_test, mock_predictions[:, 0])
        self.assertAlmostEqual(results["MSE"], expected_mse, places=4)
        self.assertIn(
            "CRPS", results
        )  # Check for existence but not the exact value in this test

    def test_poisson_metrics(self):
        # Mock predictions for Poisson
        mock_predictions = np.random.poisson(lam=3, size=100) + 1e-3
        self.model.predict = MagicMock(return_value=mock_predictions)

        self.model.get_default_metrics = MagicMock(
            return_value={"Poisson Deviance": mean_poisson_deviance}
        )

        results = self.model.evaluate(
            self.X_test, self.y_test, distribution_family="poisson"
        )
        self.assertIn("Poisson Deviance", results)
        # Optionally calculate expected deviance and check
        expected_deviance = mean_poisson_deviance(self.y_test, mock_predictions)
        self.assertAlmostEqual(results["Poisson Deviance"], expected_deviance)


if __name__ == "__main__":
    unittest.main()
