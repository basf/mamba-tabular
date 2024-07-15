import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, SplineTransformer
import numpy as np


class ScaledPolynomialLayer(nn.Module):
    def __init__(self, degree=2):
        super(ScaledPolynomialLayer, self).__init__()
        self.degree = degree

        # Initialize polynomial feature generator
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        # Initialize learnable scaling parameter
        self.weights = nn.Parameter(torch.ones(self.degree))

    def forward(self, x):
        # Scale the input to the range [-1, 1]
        x_np = x.detach().cpu().numpy()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_scaled = scaler.fit_transform(x_np) * 1e-05

        # Generate polynomial features
        poly_features = self.poly.fit_transform(x_scaled)

        # Convert polynomial features back to tensor
        poly_features = torch.tensor(poly_features, dtype=torch.float32).to(x.device)

        # Apply the learnable scaling parameter
        output = poly_features * self.weights

        output = torch.clamp(output, min=-1e5, max=1e3)

        return output


class ScaledSplineLayer(nn.Module):
    def __init__(self, degree=3, knots=63, learn_knots=True, learn_weights=True):
        super(ScaledSplineLayer, self).__init__()
        self.degree = degree
        self.knots = knots
        self.learn_knots = learn_knots
        self.learn_weights = learn_weights

        # Initialize polynomial feature generator
        self.spline = SplineTransformer(
            degree=self.degree, include_bias=False, knots=self.knots
        )

        if self.learn_knots:
            # Learnable knots
            self.knots_positions = nn.Parameter(torch.linspace(-1, 1, self.knots))
        else:
            self.knots_positions = torch.linspace(-1, 1, self.knots)

        if self.learn_weights:
            # Learnable weights for each dimension
            self.weights = nn.Parameter(torch.ones(self.knots + 1))
        else:
            self.weights = torch.ones(self.knots + 1)

    def forward(self, x):
        # Scale the input to the range [-1, 1]
        x_np = x.detach().cpu().numpy()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_scaled = scaler.fit_transform(x_np).reshape(-1, 1)

        if self.learn_knots:
            # Use learnable knots positions and ensure they are sorted
            knots = self.knots_positions.detach().cpu().numpy().reshape(-1, 1)
            sorted_knots = np.sort(knots)
            self.spline.knots = np.interp(
                sorted_knots, (sorted_knots.min(), sorted_knots.max()), (0, 1)
            )

        # Generate spline features
        spline_features = self.spline.fit_transform(x_scaled)

        # Convert spline features back to tensor
        spline_features = torch.tensor(spline_features, dtype=torch.float32).to(
            x.device
        )

        # Apply the learnable scaling parameter and weights
        output = spline_features * self.weights

        return output
