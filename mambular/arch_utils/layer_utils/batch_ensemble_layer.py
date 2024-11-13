import torch
import torch.nn as nn
from typing import Literal
import math
from typing import Callable


class LinearBatchEnsembleLayer(nn.Module):
    """
    A configurable BatchEnsemble layer that supports optional input scaling, output scaling,
    and output bias terms as per the 'BatchEnsemble' paper.
    It provides initialization options for scaling terms to diversify ensemble members.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        ensemble_scaling_in: bool = True,
        ensemble_scaling_out: bool = True,
        ensemble_bias: bool = False,
        scaling_init: Literal["ones", "random-signs"] = "ones",
    ):
        super(LinearBatchEnsembleLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        # Base weight matrix W, shared across ensemble members
        self.W = nn.Parameter(torch.randn(out_features, in_features))

        # Optional scaling factors and shifts for each ensemble member
        self.r = (
            nn.Parameter(torch.empty(ensemble_size, in_features))
            if ensemble_scaling_in
            else None
        )
        self.s = (
            nn.Parameter(torch.empty(ensemble_size, out_features))
            if ensemble_scaling_out
            else None
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features))
            if not ensemble_bias and out_features > 0
            else (
                nn.Parameter(torch.empty(ensemble_size, out_features))
                if ensemble_bias
                else None
            )
        )

        # Initialize parameters
        self.reset_parameters(scaling_init)

    def reset_parameters(self, scaling_init: Literal["ones", "random-signs", "normal"]):
        # Initialize W using a uniform distribution
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        # Initialize scaling factors r and s based on selected initialization
        scaling_init_fn = {
            "ones": nn.init.ones_,
            "random-signs": lambda x: torch.sign(torch.randn_like(x)),
            "normal": lambda x: nn.init.normal_(x, mean=0.0, std=1.0),
        }

        if self.r is not None:
            scaling_init_fn[scaling_init](self.r)
        if self.s is not None:
            scaling_init_fn[scaling_init](self.s)

        # Initialize bias
        if self.bias is not None:
            if self.bias.shape == (self.out_features,):
                nn.init.uniform_(self.bias, -0.1, 0.1)
            else:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(
                -1, self.ensemble_size, -1
            )  # Shape: (B, n_ensembles, N)
        elif x.size(1) != self.ensemble_size:
            raise ValueError(
                f"Input shape {x.shape} is invalid. Expected shape: (B, n_ensembles, N)"
            )

        # Apply input scaling if enabled
        if self.r is not None:
            x = x * self.r

        # Linear transformation with W
        output = torch.einsum("bki,oi->bko", x, self.W)

        # Apply output scaling if enabled
        if self.s is not None:
            output = output * self.s

        # Add bias if enabled
        if self.bias is not None:
            output = output + self.bias

        return output


class RNNBatchEnsembleLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ensemble_size: int,
        nonlinearity: Callable = torch.tanh,
        dropout: float = 0.0,
        ensemble_scaling_in: bool = True,
        ensemble_scaling_out: bool = True,
        ensemble_bias: bool = False,
        scaling_init: Literal["ones", "random-signs", "normal"] = "ones",
    ):
        """
        A batch ensemble RNN layer with optional bidirectionality and shared weights.

        Parameters
        ----------
        input_size : int
            The number of input features.
        hidden_size : int
            The number of features in the hidden state.
        ensemble_size : int
            The number of ensemble members.
        nonlinearity : Callable, default=torch.tanh
            Activation function to apply after each RNN step.
        dropout : float, default=0.0
            Dropout rate applied to the hidden state.
        ensemble_scaling_in : bool, default=True
            Whether to use input scaling for each ensemble member.
        ensemble_scaling_out : bool, default=True
            Whether to use output scaling for each ensemble member.
        ensemble_bias : bool, default=False
            Whether to use a unique bias term for each ensemble member.
        """
        super(RNNBatchEnsembleLayer, self).__init__()
        self.input_size = input_size
        self.ensemble_size = ensemble_size
        self.nonlinearity = nonlinearity
        self.dropout_layer = nn.Dropout(dropout)
        self.bidirectional = False
        self.num_directions = 1
        self.hidden_size = hidden_size

        # Shared RNN weight matrices for all ensemble members
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Ensemble-specific scaling factors and bias for each ensemble member
        self.r = (
            nn.Parameter(torch.empty(ensemble_size, input_size))
            if ensemble_scaling_in
            else None
        )
        self.s = (
            nn.Parameter(torch.empty(ensemble_size, hidden_size))
            if ensemble_scaling_out
            else None
        )
        self.bias = (
            nn.Parameter(torch.zeros(ensemble_size, hidden_size))
            if ensemble_bias
            else None
        )

        # Initialize parameters
        self.reset_parameters(scaling_init)

    def reset_parameters(self, scaling_init: Literal["ones", "random-signs", "normal"]):
        # Initialize scaling factors r and s based on selected initialization
        scaling_init_fn = {
            "ones": nn.init.ones_,
            "random-signs": lambda x: torch.sign(torch.randn_like(x)),
            "normal": lambda x: nn.init.normal_(x, mean=0.0, std=1.0),
        }

        if self.r is not None:
            scaling_init_fn[scaling_init](self.r)
        if self.s is not None:
            scaling_init_fn[scaling_init](self.s)

        # Xavier initialization for W_ih and W_hh like a standard RNN
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)

        # Initialize bias to zeros if applicable
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the BatchEnsembleRNNLayer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).
        hidden : torch.Tensor, optional
            Hidden state tensor of shape (num_directions, ensemble_size, batch_size, hidden_size), by default None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, ensemble_size, hidden_size * num_directions).
        """
        # Check input shape and expand if necessary
        if x.dim() == 3:  # Case: (B, L, D) - no ensembles
            batch_size, seq_len, input_size = x.shape
            x = x.unsqueeze(2).expand(
                -1, -1, self.ensemble_size, -1
            )  # Shape: (B, L, ensemble_size, D)
        elif (
            x.dim() == 4 and x.size(2) == self.ensemble_size
        ):  # Case: (B, L, ensemble_size, D)
            batch_size, seq_len, ensemble_size, _ = x.shape
            if ensemble_size != self.ensemble_size:
                raise ValueError(
                    f"Input shape {x.shape} is invalid. Expected shape: (B, S, ensemble_size, N)"
                )
        else:
            raise ValueError(
                f"Input shape {x.shape} is invalid. Expected shape: (B, L, D) or (B, L, ensemble_size, D)"
            )

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                self.num_directions,
                self.ensemble_size,
                batch_size,
                self.hidden_size,
                device=x.device,
            )

        outputs = []

        for t in range(seq_len):
            hidden_next_directions = []

            for direction in range(self.num_directions):
                # Select forward or backward timestep `t`

                t_index = t if direction == 0 else seq_len - 1 - t
                x_t = x[:, t_index, :, :]

                # Apply input scaling if enabled
                if self.r is not None:
                    x_t = x_t * self.r

                # Input and hidden term calculations with shared weights
                input_term = torch.einsum("bki,hi->bkh", x_t, self.W_ih)
                # Access the hidden state for the current direction, reshape for matrix multiplication
                hidden_direction = hidden[direction]  # Shape: (E, B, hidden_size)
                hidden_direction = hidden_direction.permute(
                    1, 0, 2
                )  # Shape: (B, E, hidden_size)
                hidden_term = torch.einsum(
                    "bki,hi->bkh", hidden_direction, self.W_hh
                )  # Shape: (B, E, hidden_size)
                hidden_next = input_term + hidden_term

                # Apply output scaling, bias, and non-linearity
                if self.s is not None:
                    hidden_next = hidden_next * self.s
                if self.bias is not None:
                    hidden_next = hidden_next + self.bias

                hidden_next = self.nonlinearity(hidden_next)
                hidden_next = hidden_next.permute(1, 0, 2)

                hidden_next_directions.append(hidden_next)

            # Stack `hidden_next_directions` along the first dimension to update `hidden` for all directions
            hidden = torch.stack(
                hidden_next_directions, dim=0
            )  # Shape: (num_directions, ensemble_size, batch_size, hidden_size)

            # Concatenate outputs for both directions along the last dimension if bidirectional
            output = torch.cat(
                [hn.permute(1, 0, 2) for hn in hidden_next_directions], dim=-1
            )  # Shape: (batch_size, ensemble_size, hidden_size * num_directions)
            outputs.append(output)

        # Apply dropout only to the final layer output if dropout is set
        if self.dropout_layer is not None:
            outputs[-1] = self.dropout_layer(outputs[-1])

        # Stack outputs for all timesteps
        outputs = torch.stack(
            outputs, dim=1
        )  # Shape: (batch_size, seq_len, ensemble_size, hidden_size * num_directions)

        return outputs, hidden
