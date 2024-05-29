import unittest
import torch
from mambular.utils.distributions import (
    NormalDistribution,
    PoissonDistribution,
    BetaDistribution,
    InverseGammaDistribution,
    DirichletDistribution,
    GammaDistribution,
    NegativeBinomialDistribution,
    CategoricalDistribution,
    StudentTDistribution,
)


class TestNormalDistribution(unittest.TestCase):
    def setUp(self):
        """Initialize the NormalDistribution object with default transforms."""
        self.normal = NormalDistribution()

    def test_initialization(self):
        """Test the initialization and default parameter settings."""
        self.assertEqual(self.normal._name, "Normal")
        self.assertEqual(self.normal.param_names, ["mean", "variance"])
        self.assertIsInstance(self.normal.mean_transform, type(lambda x: x))
        self.assertIsInstance(
            self.normal.var_transform, type(torch.nn.functional.softplus)
        )

    def test_predefined_transforms(self):
        """Test if predefined transformations are correctly applied."""
        x = torch.tensor([-1.0, 0.0, 1.0])
        self.assertTrue(
            torch.allclose(self.normal.mean_transform(x), x)
        )  # 'none' should change nothing
        self.assertTrue(
            torch.all(torch.ge(self.normal.var_transform(x), 0))
        )  # 'positive' should make all values non-negative

    def test_compute_loss_known_values(self):
        """Test the loss computation against known values."""
        predictions = torch.tensor([[0.0, 1.0]])  # mean = 0, variance = 1
        y_true = torch.tensor([0.0])
        self.normal = NormalDistribution()
        loss = self.normal.compute_loss(predictions, y_true)
        test_dist = torch.distributions.Normal(
            loc=predictions[:, 0], scale=torch.nn.functional.softplus(predictions[:, 1])
        )
        expected_loss = -test_dist.log_prob(torch.tensor(0.0)).mean()
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)

    def test_evaluate_nll(self):
        """Test the evaluate NLL function."""
        y_true = [0.0]
        y_pred = [[0.0, 1.0]]  # mean=0, variance=1
        result = self.normal.evaluate_nll(y_true, y_pred)
        self.assertIn("NLL", result)
        self.assertIn("mse", result)
        self.assertIn("mae", result)
        self.assertIn("rmse", result)


class TestPoissonDistribution(unittest.TestCase):
    def setUp(self):
        """Initialize the PoissonDistribution object with default transform."""
        self.poisson = PoissonDistribution()

    def test_initialization(self):
        """Test the initialization and parameter settings."""
        self.assertEqual(self.poisson._name, "Poisson")
        self.assertEqual(self.poisson.param_names, ["rate"])
        self.assertIsInstance(
            self.poisson.rate_transform, type(torch.nn.functional.softplus)
        )

    def test_compute_loss_known_values(self):
        """Test the loss computation against known values."""
        predictions = torch.tensor([[1.0]])  # rate = 1
        y_true = torch.tensor([1.0])
        loss = self.poisson.compute_loss(predictions, y_true)
        expected_loss = (
            -torch.distributions.Poisson(
                torch.nn.functional.softplus(predictions[:, 0])
            )
            .log_prob(torch.tensor(1.0))
            .mean()
        )
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)


class TestBetaDistribution(unittest.TestCase):
    def setUp(self):
        """Initialize the BetaDistribution object with default transforms."""
        self.beta = BetaDistribution()

    def test_initialization(self):
        """Test the initialization and parameter settings."""
        self.assertEqual(self.beta._name, "Beta")
        self.assertEqual(self.beta.param_names, ["alpha", "beta"])
        self.assertIsInstance(
            self.beta.alpha_transform, type(torch.nn.functional.softplus)
        )
        self.assertIsInstance(
            self.beta.beta_transform, type(torch.nn.functional.softplus)
        )

    def test_compute_loss_known_values(self):
        """Test the loss computation against known values."""
        predictions = torch.tensor(
            [[1.0, 1.0]]
        )  # alpha = 1, beta = 1 (uniform distribution)
        y_true = torch.tensor([0.5])
        loss = self.beta.compute_loss(predictions, y_true)
        expected_loss = (
            -torch.distributions.Beta(
                torch.nn.functional.softplus(predictions[:, 0]),
                torch.nn.functional.softplus(predictions[:, 1]),
            )
            .log_prob(torch.tensor(0.5))
            .mean()
        )
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)


class TestInverseGammaDistribution(unittest.TestCase):
    def setUp(self):
        """Initialize the InverseGammaDistribution object with default transforms."""
        self.inverse_gamma = InverseGammaDistribution()

    def test_initialization(self):
        """Test the initialization and parameter settings."""
        self.assertEqual(self.inverse_gamma._name, "InverseGamma")
        self.assertEqual(self.inverse_gamma.param_names, ["shape", "scale"])
        self.assertIsInstance(
            self.inverse_gamma.shape_transform, type(torch.nn.functional.softplus)
        )
        self.assertIsInstance(
            self.inverse_gamma.scale_transform, type(torch.nn.functional.softplus)
        )

    def test_compute_loss_known_values(self):
        """Test the loss computation against known values."""
        # These values for shape and scale parameters are chosen to be feasible and testable.
        predictions = torch.tensor([[3.0, 2.0]])  # shape = 3, scale = 2
        y_true = torch.tensor([0.5])

        loss = self.inverse_gamma.compute_loss(predictions, y_true)
        # Manually calculate the expected loss using torch's distribution functions
        shape = torch.nn.functional.softplus(predictions[:, 0])
        scale = torch.nn.functional.softplus(predictions[:, 1])
        inverse_gamma_dist = torch.distributions.InverseGamma(shape, scale)
        expected_loss = -inverse_gamma_dist.log_prob(y_true).mean()

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)


class TestDirichletDistribution(unittest.TestCase):
    def setUp(self):
        """Initialize the DirichletDistribution object with default transforms."""
        self.dirichlet = DirichletDistribution()

    def test_initialization(self):
        """Test the initialization and parameter settings."""
        self.assertEqual(self.dirichlet._name, "Dirichlet")
        # Concentration param_name is a simplification as mentioned in your class docstring
        self.assertEqual(self.dirichlet.param_names, ["concentration"])
        self.assertIsInstance(
            self.dirichlet.concentration_transform, type(torch.nn.functional.softplus)
        )

    def test_compute_loss_known_values(self):
        """Test the loss computation against known values."""
        # These values are chosen to be feasible and testable.
        # Example: Concentrations for a 3-dimensional Dirichlet distribution
        predictions = torch.tensor(
            [[1.0, 1.0, 1.0]]
        )  # Equal concentration, should resemble uniform distribution over simplex
        y_true = torch.tensor(
            [[0.33, 0.33, 0.34]]
        )  # Example point in the probability simplex

        loss = self.dirichlet.compute_loss(predictions, y_true)
        # Manually calculate the expected loss using torch's distribution functions
        concentration = torch.nn.functional.softplus(predictions)
        dirichlet_dist = torch.distributions.Dirichlet(concentration)
        expected_loss = -dirichlet_dist.log_prob(y_true).mean()

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)


class TestGammaDistribution(unittest.TestCase):
    def setUp(self):
        """Initialize the GammaDistribution object with default transforms."""
        self.gamma = GammaDistribution()

    def test_initialization(self):
        """Test the initialization and parameter settings."""
        self.assertEqual(self.gamma._name, "Gamma")
        self.assertEqual(self.gamma.param_names, ["shape", "rate"])
        self.assertIsInstance(
            self.gamma.shape_transform, type(torch.nn.functional.softplus)
        )
        self.assertIsInstance(
            self.gamma.rate_transform, type(torch.nn.functional.softplus)
        )

    def test_compute_loss_known_values(self):
        """Test the loss computation against known values."""
        # Set some test parameters and observations
        predictions = torch.tensor([[2.0, 3.0]])  # shape = 2, rate = 3
        y_true = torch.tensor([0.5])  # Test value

        loss = self.gamma.compute_loss(predictions, y_true)
        # Manually calculate the expected loss using torch's distribution functions
        shape = torch.nn.functional.softplus(predictions[:, 0])
        rate = torch.nn.functional.softplus(predictions[:, 1])
        gamma_dist = torch.distributions.Gamma(shape, rate)
        expected_loss = -gamma_dist.log_prob(y_true).mean()

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)


class TestStudentTDistribution(unittest.TestCase):
    def setUp(self):
        """Initialize the StudentTDistribution object with default transforms."""
        self.student_t = StudentTDistribution()

    def test_initialization(self):
        """Test the initialization and parameter settings."""
        self.assertEqual(self.student_t._name, "StudentT")
        self.assertEqual(self.student_t.param_names, ["df", "loc", "scale"])
        self.assertIsInstance(
            self.student_t.df_transform, type(torch.nn.functional.softplus)
        )
        self.assertIsInstance(
            self.student_t.loc_transform,
            type(lambda x: x),  # Assuming 'none' transformation
        )
        self.assertIsInstance(
            self.student_t.scale_transform, type(torch.nn.functional.softplus)
        )

    def test_compute_loss_known_values(self):
        """Test the loss computation against known values."""
        # Set some test parameters and observations
        predictions = torch.tensor([[10.0, 0.0, 1.0]])  # df=10, loc=0, scale=1
        y_true = torch.tensor([0.5])  # Test value

        loss = self.student_t.compute_loss(predictions, y_true)
        # Manually calculate the expected loss using torch's distribution functions
        df = torch.nn.functional.softplus(predictions[:, 0])
        loc = predictions[:, 1]  # 'none' transformation
        scale = torch.nn.functional.softplus(predictions[:, 2])
        student_t_dist = torch.distributions.StudentT(df, loc, scale)
        expected_loss = -student_t_dist.log_prob(y_true).mean()

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)

    def test_evaluate_nll(self):
        """Test the evaluate NLL function and additional metrics."""
        y_true = [0.5]
        y_pred = [[10.0, 0.0, 1.0]]  # df=10, loc=0, scale=1
        result = self.student_t.evaluate_nll(y_true, y_pred)

        self.assertIn("NLL", result)
        self.assertIn("mse", result)
        self.assertIn("mae", result)
        self.assertIn("rmse", result)

        # Check that MSE, MAE, RMSE calculations are reasonable
        self.assertGreaterEqual(result["mse"], 0)
        self.assertGreaterEqual(result["mae"], 0)
        self.assertGreaterEqual(result["rmse"], 0)


class TestNegativeBinomialDistribution(unittest.TestCase):
    def setUp(self):
        """Initialize the NegativeBinomialDistribution object with default transforms."""
        self.negative_binomial = NegativeBinomialDistribution()

    def test_initialization(self):
        """Test the initialization and parameter settings."""
        self.assertEqual(self.negative_binomial._name, "NegativeBinomial")
        self.assertEqual(self.negative_binomial.param_names, ["mean", "dispersion"])
        self.assertIsInstance(
            self.negative_binomial.mean_transform, type(torch.nn.functional.softplus)
        )
        self.assertIsInstance(
            self.negative_binomial.dispersion_transform,
            type(torch.nn.functional.softplus),
        )

    def test_compute_loss_known_values(self):
        """Test the loss computation against known values."""
        # Set some test parameters and observations
        predictions = torch.tensor([[10.0, 0.1]])  # mean=10, dispersion=0.1
        y_true = torch.tensor([5.0])  # Test value

        loss = self.negative_binomial.compute_loss(predictions, y_true)
        # Manually calculate the expected loss using torch's distribution functions
        mean = torch.nn.functional.softplus(predictions[:, 0])
        dispersion = torch.nn.functional.softplus(predictions[:, 1])
        r = 1 / dispersion
        p = r / (r + mean)
        negative_binomial_dist = torch.distributions.NegativeBinomial(
            total_count=r, probs=p
        )
        expected_loss = -negative_binomial_dist.log_prob(y_true).mean()

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)


class TestCategoricalDistribution(unittest.TestCase):
    def setUp(self):
        """Initialize the CategoricalDistribution object with a probability transformation."""
        self.categorical = CategoricalDistribution()

    def test_initialization(self):
        """Test the initialization and parameter settings."""
        self.assertEqual(self.categorical._name, "Categorical")
        self.assertEqual(self.categorical.param_names, ["probs"])
        # The transformation function will need to ensure the probabilities are valid (non-negative and sum to 1)
        # Typically, this might involve applying softmax to ensure the constraints are met.
        # Here, we assume `prob_transform` is something akin to softmax for the sake of test setup.
        self.assertIsInstance(
            self.categorical.probs_transform, type(torch.nn.functional.softmax)
        )

    def test_compute_loss_known_values(self):
        # Example with three categories
        logits = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 3.0, 4.0]]
        )  # Logits for three categories
        y_true = torch.tensor([2, 1])

        loss = self.categorical.compute_loss(logits, y_true)
        # Apply softmax to logits to convert them into probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        cat_dist = torch.distributions.Categorical(probs=probs)
        expected_loss = -cat_dist.log_prob(y_true).mean()

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)


# Running the tests
if __name__ == "__main__":
    unittest.main()
