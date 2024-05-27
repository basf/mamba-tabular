import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss

from mambular.models import MambularClassifier  # Ensure correct import path


class TestMambularClassifier(unittest.TestCase):
    def setUp(self):
        # Patching external dependencies
        self.patcher_pl_trainer = patch("lightning.Trainer")
        self.mock_pl_trainer = self.patcher_pl_trainer.start()

        self.patcher_base_model = patch(
            "mambular.base_models.classifier.BaseMambularClassifier"
        )
        self.mock_base_model = self.patcher_base_model.start()

        self.classifier = MambularClassifier(d_model=128, dropout=0.1)

        # Sample data
        self.X = pd.DataFrame(np.random.randn(100, 10))
        self.y = np.random.choice(["A", "B", "C"], size=100)

        self.classifier.cat_feature_info = {}
        self.classifier.num_feature_info = {}

    def tearDown(self):
        self.patcher_pl_trainer.stop()
        self.patcher_base_model.stop()

    def test_initialization(self):
        # This assumes MambularConfig is properly imported and used in the MambularRegressor class
        from mambular.utils.config import MambularConfig

        self.assertIsInstance(self.classifier.config, MambularConfig)
        self.assertEqual(self.classifier.config.d_model, 128)
        self.assertEqual(self.classifier.config.dropout, 0.1)

    def test_split_data(self):
        """Test the data splitting functionality."""
        X_train, X_val, y_train, y_val = self.classifier.split_data(
            self.X, self.y, val_size=0.2, random_state=42
        )
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_val), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_val), 20)

    def test_fit(self):
        """Test the training setup and call."""
        # Mock the necessary parts to simulate training
        self.classifier.preprocess_data = MagicMock()
        self.classifier.model = self.mock_base_model

        self.classifier.fit(self.X, self.y)

        # Ensure that the fit method of the trainer is called
        self.mock_pl_trainer.return_value.fit.assert_called_once()

    def test_predict(self):
        # Create a mock tensor as the model output
        # Assuming three classes A, B, C as per self.y
        mock_logits = torch.rand(100, 3)

        # Mock the model and its method calls
        self.classifier.model = MagicMock()
        self.classifier.model.eval.return_value = None
        self.classifier.model.return_value = mock_logits

        # Mock preprocess_test_data to return dummy tensor data
        self.classifier.preprocess_test_data = MagicMock(return_value=([], []))

        predictions = self.classifier.predict(self.X)

        # Assert that predictions return as expected
        expected_predictions = torch.argmax(mock_logits, dim=1).numpy()
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_evaluate(self):
        # Mock predict and predict_proba to simulate classifier output
        mock_predictions = np.random.choice([0, 1, 2], size=100)
        raw_probabilities = np.random.rand(100, 3)
        # Normalize these probabilities so that each row sums to 1
        mock_probabilities = raw_probabilities / raw_probabilities.sum(
            axis=1, keepdims=True
        )
        self.classifier.predict = MagicMock(return_value=mock_predictions)
        self.classifier.predict_proba = MagicMock(
            return_value=mock_probabilities)

        # Define metrics to test
        metrics = {
            "Accuracy": (accuracy_score, False),
            # Log Loss requires probability scores
            "Log Loss": (log_loss, True),
        }

        # Call evaluate with the defined metrics
        result = self.classifier.evaluate(self.X, self.y, metrics=metrics)

        # Assert that predict and predict_proba were called correctly
        self.classifier.predict.assert_called_once()
        self.classifier.predict_proba.assert_called_once()

        # Check the results of evaluate
        expected_accuracy = accuracy_score(self.y, mock_predictions)
        expected_log_loss = log_loss(self.y, mock_probabilities)
        self.assertEqual(result["Accuracy"], expected_accuracy)
        self.assertAlmostEqual(result["Log Loss"], expected_log_loss)

        # Assert calls with appropriate arguments
        self.classifier.predict.assert_called_once_with(self.X)
        self.classifier.predict_proba.assert_called_once_with(self.X)


if __name__ == "__main__":
    unittest.main()
