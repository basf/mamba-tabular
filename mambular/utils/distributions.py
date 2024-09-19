import torch
import torch.distributions as dist
import numpy as np


class BaseDistribution(torch.nn.Module):
    """
    The base class for various statistical distributions, providing a common interface and utilities.

    This class defines the basic structure and methods that are inherited by specific distribution
    classes, allowing for the implementation of custom distributions with specific parameter transformations
    and loss computations.

    Attributes:
        _name (str): The name of the distribution.
        param_names (list of str): A list of names for the parameters of the distribution.
        param_count (int): The number of parameters for the distribution.
        predefined_transforms (dict): A dictionary of predefined transformation functions for parameters.

    Parameters:
        name (str): The name of the distribution.
        param_names (list of str): A list of names for the parameters of the distribution.
    """

    def __init__(self, name, param_names):
        super(BaseDistribution, self).__init__()

        self._name = name
        self.param_names = param_names
        self.param_count = len(param_names)
        # Predefined transformation functions accessible to all subclasses
        self.predefined_transforms = {
            "positive": torch.nn.functional.softplus,
            "none": lambda x: x,
            "square": lambda x: x**2,
            "exp": torch.exp,
            "sqrt": torch.sqrt,
            "probabilities": lambda x: torch.softmax(x, dim=-1),
            "log": lambda x: torch.log(
                x + 1e-6
            ),  # Adding a small constant for numerical stability
        }

    @property
    def name(self):
        return self._name

    @property
    def parameter_count(self):
        return self.param_count

    def get_transform(self, transform_name):
        """
        Retrieve a transformation function by name, or return the function if it's custom.
        """
        if callable(transform_name):
            # Custom transformation function provided
            return transform_name
        return self.predefined_transforms.get(
            transform_name, lambda x: x
        )  # Default to 'none'

    def compute_loss(self, predictions, y_true):
        """
        Computes the loss (e.g., negative log likelihood) for the distribution given predictions and true values.

        This method must be implemented by subclasses.

        Parameters:
            predictions (torch.Tensor): The predicted parameters of the distribution.
            y_true (torch.Tensor): The true values.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate_nll(self, y_true, y_pred):
        """
        Evaluates the negative log likelihood (NLL) for given true values and predictions.

        Parameters:
            y_true (array-like): The true values.
            y_pred (array-like): The predicted values.

        Returns:
            dict: A dictionary containing the NLL value.
        """

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        # Compute NLL using the provided loss function
        nll_loss_tensor = self.compute_loss(y_pred_tensor, y_true_tensor)

        # Convert the NLL loss tensor back to a numpy array and return
        return {
            "NLL": nll_loss_tensor.detach().numpy(),
        }

    def forward(self, predictions):
        """
        Apply the appropriate transformations to the predicted parameters.

        Parameters:
            predictions (torch.Tensor): The predicted parameters of the distribution.

        Returns:
            torch.Tensor: A tensor with transformed parameters.
        """
        transformed_params = []
        for idx, param_name in enumerate(self.param_names):
            transform_func = self.get_transform(
                getattr(self, f"{param_name}_transform", "none")
            )
            transformed_params.append(transform_func(predictions[:, idx]).unsqueeze(1))
        return torch.cat(transformed_params, dim=1)


class NormalDistribution(BaseDistribution):
    """
    Represents a Normal (Gaussian) distribution with parameters for mean and variance, including functionality
    for transforming these parameters and computing the loss.

    Inherits from BaseDistribution.

    Parameters:
        name (str): The name of the distribution. Defaults to "Normal".
        mean_transform (str or callable): The transformation for the mean parameter. Defaults to "none".
        var_transform (str or callable): The transformation for the variance parameter. Defaults to "positive".
    """

    def __init__(self, name="Normal", mean_transform="none", var_transform="positive"):
        param_names = [
            "mean",
            "variance",
        ]
        super().__init__(name, param_names)

        self.mean_transform = self.get_transform(mean_transform)
        self.variance_transform = self.get_transform(var_transform)

    def compute_loss(self, predictions, y_true):
        mean = self.mean_transform(predictions[:, self.param_names.index("mean")])
        variance = self.variance_transform(
            predictions[:, self.param_names.index("variance")]
        )

        normal_dist = dist.Normal(mean, variance)

        nll = -normal_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        mse_loss = torch.nn.functional.mse_loss(
            y_true_tensor, y_pred_tensor[:, self.param_names.index("mean")]
        )
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = (
            torch.nn.functional.l1_loss(
                y_true_tensor, y_pred_tensor[:, self.param_names.index("mean")]
            )
            .detach()
            .numpy()
        )

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        # Convert the NLL loss tensor back to a numpy array and return
        return metrics


class PoissonDistribution(BaseDistribution):
    """
    Represents a Poisson distribution, typically used for modeling count data or the number of events
    occurring within a fixed interval of time or space. This class extends the BaseDistribution and
    includes parameter transformation and loss computation specific to the Poisson distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Poisson".
        rate_transform (str or callable): Transformation to apply to the rate parameter to ensure it remains positive.
    """

    def __init__(self, name="Poisson", rate_transform="positive"):
        param_names = ["rate"]  # Specify parameter name for Poisson distribution
        super().__init__(name, param_names)
        # Retrieve transformation function for rate
        self.rate_transform = self.get_transform(rate_transform)

    def compute_loss(self, predictions, y_true):
        rate = self.rate_transform(predictions[:, self.param_names.index("rate")])

        # Define the Poisson distribution with the transformed parameter
        poisson_dist = dist.Poisson(rate)

        # Compute the negative log-likelihood
        nll = -poisson_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        rate = self.rate_transform(y_pred_tensor[:, self.param_names.index("rate")])

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, rate)
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = torch.nn.functional.l1_loss(y_true_tensor, rate).detach().numpy()
        poisson_deviance = 2 * torch.sum(
            y_true_tensor * torch.log(y_true_tensor / rate) - (y_true_tensor - rate)
        )

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse
        metrics["poisson_deviance"] = poisson_deviance.detach().numpy()

        # Convert the NLL loss tensor back to a numpy array and return
        return metrics


class InverseGammaDistribution(BaseDistribution):
    """
    Represents an Inverse Gamma distribution, often used as a prior distribution in Bayesian statistics,
    especially for scale parameters in other distributions. This class extends BaseDistribution and includes
    parameter transformation and loss computation specific to the Inverse Gamma distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "InverseGamma".
        shape_transform (str or callable): Transformation for the shape parameter to ensure it remains positive.
        scale_transform (str or callable): Transformation for the scale parameter to ensure it remains positive.
    """

    def __init__(
        self,
        name="InverseGamma",
        shape_transform="positive",
        scale_transform="positive",
    ):
        param_names = [
            "shape",
            "scale",
        ]
        super().__init__(name, param_names)

        self.shape_transform = self.get_transform(shape_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        shape = self.shape_transform(predictions[:, self.param_names.index("shape")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        inverse_gamma_dist = dist.InverseGamma(shape, scale)
        # Compute the negative log-likelihood
        nll = -inverse_gamma_dist.log_prob(y_true).mean()
        return nll


class BetaDistribution(BaseDistribution):
    """
    Represents a Beta distribution, a continuous distribution defined on the interval [0, 1], commonly used
    in Bayesian statistics for modeling probabilities. This class extends BaseDistribution and includes parameter
    transformation and loss computation specific to the Beta distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Beta".
        shape_transform (str or callable): Transformation for the alpha (shape) parameter to ensure it remains positive.
        scale_transform (str or callable): Transformation for the beta (scale) parameter to ensure it remains positive.
    """

    def __init__(
        self,
        name="Beta",
        shape_transform="positive",
        scale_transform="positive",
    ):
        param_names = [
            "alpha",
            "beta",
        ]
        super().__init__(name, param_names)

        self.alpha_transform = self.get_transform(shape_transform)
        self.beta_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        alpha = self.alpha_transform(predictions[:, self.param_names.index("alpha")])
        beta = self.beta_transform(predictions[:, self.param_names.index("beta")])

        beta_dist = dist.Beta(alpha, beta)
        # Compute the negative log-likelihood
        nll = -beta_dist.log_prob(y_true).mean()
        return nll


class DirichletDistribution(BaseDistribution):
    """
    Represents a Dirichlet distribution, a multivariate generalization of the Beta distribution. It is commonly
    used in Bayesian statistics for modeling multinomial distribution probabilities. This class extends
    BaseDistribution and includes parameter transformation and loss computation specific to the Dirichlet distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Dirichlet".
        concentration_transform (str or callable): Transformation to apply to concentration parameters to ensure they remain positive.
    """

    def __init__(self, name="Dirichlet", concentration_transform="positive"):
        # For Dirichlet, param_names could be dynamically set based on the dimensionality of alpha
        # For simplicity, we're not specifying individual names for each concentration parameter
        param_names = ["concentration"]  # This is a simplification
        super().__init__(name, param_names)
        # Retrieve transformation function for concentration parameters
        self.concentration_transform = self.get_transform(concentration_transform)

    def compute_loss(self, predictions, y_true):
        # Apply the transformation to ensure all concentration parameters are positive
        # Assuming predictions is a 2D tensor where each row is a set of concentration parameters for a Dirichlet distribution
        concentration = self.concentration_transform(predictions)

        dirichlet_dist = dist.Dirichlet(concentration)

        nll = -dirichlet_dist.log_prob(y_true).mean()
        return nll


class GammaDistribution(BaseDistribution):
    """
    Represents a Gamma distribution, a two-parameter family of continuous probability distributions. It's
    widely used in various fields of science for modeling a wide range of phenomena. This class extends
    BaseDistribution and includes parameter transformation and loss computation specific to the Gamma distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Gamma".
        shape_transform (str or callable): Transformation for the shape parameter to ensure it remains positive.
        rate_transform (str or callable): Transformation for the rate parameter to ensure it remains positive.
    """

    def __init__(
        self, name="Gamma", shape_transform="positive", rate_transform="positive"
    ):
        param_names = ["shape", "rate"]
        super().__init__(name, param_names)

        self.shape_transform = self.get_transform(shape_transform)
        self.rate_transform = self.get_transform(rate_transform)

    def compute_loss(self, predictions, y_true):
        shape = self.shape_transform(predictions[:, self.param_names.index("shape")])
        rate = self.rate_transform(predictions[:, self.param_names.index("rate")])

        # Define the Gamma distribution with the transformed parameters
        gamma_dist = dist.Gamma(shape, rate)

        # Compute the negative log-likelihood
        nll = -gamma_dist.log_prob(y_true).mean()
        return nll


class StudentTDistribution(BaseDistribution):
    """
    Represents a Student's t-distribution, a family of continuous probability distributions that arise when
    estimating the mean of a normally distributed population in situations where the sample size is small.
    This class extends BaseDistribution and includes parameter transformation and loss computation specific
    to the Student's t-distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "StudentT".
        df_transform (str or callable): Transformation for the degrees of freedom parameter to ensure it remains positive.
        loc_transform (str or callable): Transformation for the location parameter.
        scale_transform (str or callable): Transformation for the scale parameter to ensure it remains positive.
    """

    def __init__(
        self,
        name="StudentT",
        df_transform="positive",
        loc_transform="none",
        scale_transform="positive",
    ):
        param_names = ["df", "loc", "scale"]
        super().__init__(name, param_names)

        self.df_transform = self.get_transform(df_transform)
        self.loc_transform = self.get_transform(loc_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        df = self.df_transform(predictions[:, self.param_names.index("df")])
        loc = self.loc_transform(predictions[:, self.param_names.index("loc")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        student_t_dist = dist.StudentT(df, loc, scale)

        nll = -student_t_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        mse_loss = torch.nn.functional.mse_loss(
            y_true_tensor, y_pred_tensor[:, self.param_names.index("loc")]
        )
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = (
            torch.nn.functional.l1_loss(
                y_true_tensor, y_pred_tensor[:, self.param_names.index("loc")]
            )
            .detach()
            .numpy()
        )

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        # Convert the NLL loss tensor back to a numpy array and return
        return metrics


class NegativeBinomialDistribution(BaseDistribution):
    """
    Represents a Negative Binomial distribution, often used for count data and modeling the number of failures
    before a specified number of successes occurs in a series of Bernoulli trials. This class extends
    BaseDistribution and includes parameter transformation and loss computation specific to the Negative Binomial distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "NegativeBinomial".
        mean_transform (str or callable): Transformation for the mean parameter to ensure it remains positive.
        dispersion_transform (str or callable): Transformation for the dispersion parameter to ensure it remains positive.
    """

    def __init__(
        self,
        name="NegativeBinomial",
        mean_transform="positive",
        dispersion_transform="positive",
    ):
        param_names = ["mean", "dispersion"]
        super().__init__(name, param_names)

        self.mean_transform = self.get_transform(mean_transform)
        self.dispersion_transform = self.get_transform(dispersion_transform)

    def compute_loss(self, predictions, y_true):
        # Apply transformations to ensure mean and dispersion parameters are positive
        mean = self.mean_transform(predictions[:, self.param_names.index("mean")])
        dispersion = self.dispersion_transform(
            predictions[:, self.param_names.index("dispersion")]
        )

        # Calculate the probability (p) and number of successes (r) from mean and dispersion
        # These calculations follow from the mean and variance of the negative binomial distribution
        # where variance = mean + mean^2 / dispersion
        r = 1 / dispersion
        p = r / (r + mean)

        # Define the Negative Binomial distribution with the transformed parameters
        negative_binomial_dist = dist.NegativeBinomial(total_count=r, probs=p)

        # Compute the negative log-likelihood
        nll = -negative_binomial_dist.log_prob(y_true).mean()
        return nll


class CategoricalDistribution(BaseDistribution):
    """
    Represents a Categorical distribution, a discrete distribution that describes the possible results of a
    random variable that can take on one of K possible categories, with the probability of each category
    separately specified. This class extends BaseDistribution and includes parameter transformation and loss
    computation specific to the Categorical distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Categorical".
        prob_transform (str or callable): Transformation for the probabilities to ensure they remain valid (i.e., non-negative and sum to 1).
    """

    def __init__(self, name="Categorical", prob_transform="probabilities"):
        param_names = ["probs"]  # Specify parameter name for Poisson distribution
        super().__init__(name, param_names)
        # Retrieve transformation function for rate
        self.probs_transform = self.get_transform(prob_transform)

    def compute_loss(self, predictions, y_true):
        probs = self.probs_transform(predictions)

        # Define the Poisson distribution with the transformed parameter
        cat_dist = dist.Categorical(probs=probs)

        # Compute the negative log-likelihood
        nll = -cat_dist.log_prob(y_true).mean()
        return nll


class Quantile(BaseDistribution):
    """
    Quantile Regression Loss class.

    This class computes the quantile loss (also known as pinball loss) for a set of quantiles.
    It is used to handle quantile regression tasks where we aim to predict a given quantile of the target distribution.

    Parameters
    ----------
    name : str, optional
        The name of the distribution, by default "Quantile".
    quantiles : list of float, optional
        A list of quantiles to be used for computing the loss, by default [0.25, 0.5, 0.75].

    Attributes
    ----------
    quantiles : list of float
        List of quantiles for which the pinball loss is computed.

    Methods
    -------
    compute_loss(predictions, y_true)
        Computes the quantile regression loss between the predictions and true values.
    """

    def __init__(self, name="Quantile", quantiles=[0.25, 0.5, 0.75]):
        param_names = [
            f"q_{q}" for q in quantiles
        ]  # Use string representations of quantiles
        super().__init__(name, param_names)
        self.quantiles = quantiles

    def compute_loss(self, predictions, y_true):

        assert not y_true.requires_grad  # Ensure y_true does not require gradients
        assert predictions.size(0) == y_true.size(0)  # Ensure batch size matches

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = y_true - predictions[:, i]  # Calculate errors for each quantile
            # Compute the pinball loss
            quantile_loss = torch.max((q - 1) * errors, q * errors)
            losses.append(quantile_loss)

        # Sum losses across quantiles and compute mean
        loss = torch.mean(torch.stack(losses, dim=1).sum(dim=1))
        return loss
