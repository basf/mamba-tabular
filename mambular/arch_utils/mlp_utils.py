import torch.nn as nn


class Linear_skip_block(nn.Module):
    """
    A neural network block that includes a linear layer, an activation function, a dropout layer, and optionally a
    skip connection and batch normalization. The skip connection is added if the input and output feature sizes are equal.

    Parameters
    ----------
    n_input : int
        The number of input features.
    n_output : int
        The number of output features.
    dropout_rate : float
        The rate of dropout to apply for regularization.
    activation_fn : torch.nn.modules.activation, optional
        The activation function to use after the linear layer. Default is nn.LeakyReLU().
    use_batch_norm : bool, optional
        Whether to apply batch normalization after the activation function. Default is False.

    Attributes
    ----------
    fc : torch.nn.Linear
        The linear transformation layer.
    act : torch.nn.Module
        The activation function.
    drop : torch.nn.Dropout
        The dropout layer.
    use_batch_norm : bool
        Indicator of whether batch normalization is used.
    batch_norm : torch.nn.BatchNorm1d, optional
        The batch normalization layer, instantiated if use_batch_norm is True.
    use_skip : bool
        Indicator of whether a skip connection is used.
    """

    def __init__(
        self,
        n_input,
        n_output,
        dropout_rate,
        activation_fn=nn.LeakyReLU(),
        use_batch_norm=False,
    ):
        super(Linear_skip_block, self).__init__()

        self.fc = nn.Linear(n_input, n_output)
        self.act = activation_fn
        self.drop = nn.Dropout(dropout_rate)
        self.use_batch_norm = use_batch_norm
        self.use_skip = (
            n_input == n_output
        )  # Only use skip connection if input and output sizes are equal

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(n_output)  # Initialize batch normalization

    def forward(self, x):
        """
        Defines the forward pass of the Linear_block.

        Parameters
        ----------
        x : Tensor
            The input tensor to the block.

        Returns
        -------
        Tensor
            The output tensor after processing through the linear layer, activation function, dropout,
            and optional batch normalization.
        """
        x0 = x  # Save input for possible skip connection
        x = self.fc(x)
        x = self.act(x)

        if self.use_batch_norm:
            x = self.batch_norm(x)  # Apply batch normalization after activation

        if self.use_skip:
            x = x + x0  # Add skip connection if applicable

        x = self.drop(x)  # Apply dropout
        return x


class Linear_block(nn.Module):
    """
    A neural network block that includes a linear layer, an activation function, a dropout layer, and optionally batch normalization.

    Parameters
    ----------
    n_input : int
        The number of input features.
    n_output : int
        The number of output features.
    dropout_rate : float
        The rate of dropout to apply.
    activation_fn : torch.nn.modules.activation, optional
        The activation function to use after the linear layer. Default is nn.LeakyReLU().
    batch_norm : bool, optional
        Whether to include batch normalization after the activation function. Default is False.

    Attributes
    ----------
    block : torch.nn.Sequential
        A sequential container holding the linear layer, activation function, dropout, and optionally batch normalization.
    """

    def __init__(
        self,
        n_input,
        n_output,
        dropout_rate,
        activation_fn=nn.LeakyReLU(),
        batch_norm=False,
    ):
        super(Linear_block, self).__init__()

        # Initialize modules
        modules = [
            nn.Linear(n_input, n_output),
            activation_fn,
            nn.Dropout(dropout_rate),
        ]

        # Optionally add batch normalization
        if batch_norm:
            modules.append(nn.BatchNorm1d(n_output))

        # Create the sequential model
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        """
        Defines the forward pass of the Linear_block.

        Parameters
        ----------
        x : Tensor
            The input tensor to the block.

        Returns
        -------
        Tensor
            The output tensor after processing through the linear layer, activation function, dropout,
            and optional batch normalization.
        """
        # Pass the input through the block
        return self.block(x)


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) for regression tasks, configurable with optional skip connections and batch normalization.

    Parameters
    ----------
    n_input_units : int
        The number of units in the input layer.
    hidden_units_list : list of int
        A list specifying the number of units in each hidden layer.
    n_output_units : int
        The number of units in the output layer.
    dropout_rate : float
        The dropout rate used across the MLP.
    use_skip_layers : bool, optional
        Whether to use skip connections in layers where input and output sizes match. Default is False.
    activation_fn : torch.nn.modules.activation, optional
        The activation function used across the layers. Default is nn.LeakyReLU().
    use_batch_norm : bool, optional
        Whether to apply batch normalization in each layer. Default is False.

    Attributes
    ----------
    hidden_layers : torch.nn.Sequential
        Sequential container of layers comprising the MLP's hidden layers.
    linear_final : torch.nn.Linear
        The final linear layer of the MLP.
    """

    def __init__(
        self,
        n_input_units,
        hidden_units_list=[64, 32, 32],
        n_output_units: int = 1,
        dropout_rate: float = 0.1,
        use_skip_layers: bool = False,
        activation_fn=nn.LeakyReLU(),
        use_batch_norm: bool = False,
    ):
        super(MLP, self).__init__()
        self.n_input_units = n_input_units
        self.hidden_units_list = hidden_units_list
        self.dropout_rate = dropout_rate
        self.n_output_units = n_output_units

        layers = []
        input_units = n_input_units

        for n_hidden_units in hidden_units_list:
            if use_skip_layers and input_units == n_hidden_units:
                layers.append(
                    Linear_skip_block(
                        input_units,
                        n_hidden_units,
                        dropout_rate,
                        activation_fn,
                        use_batch_norm,
                    )
                )
            else:
                layers.append(
                    Linear_block(
                        input_units,
                        n_hidden_units,
                        dropout_rate,
                        activation_fn,
                        use_batch_norm,
                    )
                )
            input_units = n_hidden_units  # Update input_units for the next layer

        self.hidden_layers = nn.Sequential(*layers)
        self.linear_final = nn.Linear(input_units, n_output_units)  # Final layer

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Parameters
        ----------
        x : Tensor
            The input tensor to the MLP.

        Returns
        -------
        Tensor
            The output predictions of the model for regression tasks.
        """
        x = self.hidden_layers(x)
        x = self.linear_final(x)
        return x
