import torch
import torch.nn as nn
from .lstm_utils import mLSTMblock, sLSTMblock


class ConvRNN(nn.Module):
    def __init__(
        self,
        model_type: str,  # 'RNN', 'LSTM', or 'GRU'
        input_size: int,  # Number of input features (128 in your case)
        hidden_size: int,  # Number of hidden units in RNN layers
        num_layers: int,  # Number of RNN layers
        bidirectional: bool,  # Whether RNN is bidirectional
        rnn_dropout: float,  # Dropout rate for RNN
        bias: bool,  # Bias for RNN
        conv_bias: bool,  # Bias for Conv1d
        rnn_activation: str = None,  # Only for RNN
        d_conv: int = 4,  # Kernel size for Conv1d
        residuals: bool = False,  # Whether to use residual connections
    ):
        super(ConvRNN, self).__init__()

        # Choose RNN layer based on model_type
        rnn_layer = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "mLSTM": mLSTMblock,
            "sLSTM": sLSTMblock,
        }[model_type]

        self.input_size = input_size  # Number of input features (128 in your case)
        self.hidden_size = hidden_size  # Number of hidden units in RNN
        self.num_layers = num_layers  # Number of RNN layers
        self.bidirectional = bidirectional  # Whether RNN is bidirectional
        self.rnn_type = model_type
        self.residuals = residuals

        # Convolutional layers
        self.convs = nn.ModuleList()
        self.layernorms_conv = nn.ModuleList()  # LayerNorms for Conv layers

        if self.residuals:
            self.residual_matrix = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(hidden_size, hidden_size))
                    for _ in range(num_layers)
                ]
            )

        # First Conv1d layer uses input_size
        self.convs.append(
            nn.Conv1d(
                in_channels=self.input_size,
                out_channels=self.input_size,
                kernel_size=d_conv,
                padding=d_conv - 1,
                bias=conv_bias,
                groups=self.input_size,
            )
        )
        self.layernorms_conv.append(
            nn.LayerNorm(self.input_size)
        )  # LayerNorm for first Conv layer

        # Subsequent Conv1d layers use hidden_size as input
        for i in range(self.num_layers - 1):
            self.convs.append(
                nn.Conv1d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=d_conv,
                    padding=d_conv - 1,
                    bias=conv_bias,
                    groups=self.hidden_size,
                )
            )
            self.layernorms_conv.append(
                nn.LayerNorm(self.hidden_size)
            )  # LayerNorm for Conv layers

        # Initialize the RNN layers
        self.rnns = nn.ModuleList()
        self.layernorms_rnn = nn.ModuleList()  # LayerNorms for RNN layers

        for i in range(self.num_layers):
            if model_type in ["RNN"]:
                rnn = rnn_layer(
                    input_size=(self.input_size if i == 0 else self.hidden_size),
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    bidirectional=self.bidirectional,
                    batch_first=True,
                    dropout=rnn_dropout if i < self.num_layers - 1 else 0,
                    bias=bias,
                    nonlinearity=rnn_activation,
                )
            else:
                rnn = rnn_layer(
                    input_size=(self.input_size if i == 0 else self.hidden_size),
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    bidirectional=self.bidirectional,
                    batch_first=True,
                    dropout=rnn_dropout if i < self.num_layers - 1 else 0,
                    bias=bias,
                )
            self.rnns.append(rnn)
            self.layernorms_rnn.append(nn.LayerNorm(self.hidden_size))

    def forward(self, x):
        """
        Forward pass through Conv-RNN layers.

        Parameters
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_size).

        Returns
        --------
        output : torch.Tensor
            Output tensor after passing through Conv-RNN layers.
        """
        _, L, _ = x.shape
        if self.residuals:
            residual = x

        # Loop through the RNN layers and apply 1D convolution before each
        for i in range(self.num_layers):
            # Transpose to (batch_size, input_size, seq_length) for Conv1d
            x = self.layernorms_conv[i](x)
            x = x.transpose(1, 2)

            # Apply the 1D convolution
            x = self.convs[i](x)[:, :, :L]

            # Transpose back to (batch_size, seq_length, input_size)
            x = x.transpose(1, 2)

            # Pass through the RNN layer
            x, _ = self.rnns[i](x)

            # Residual connection with learnable matrix
            if self.residuals:
                if i < self.num_layers and i > 0:
                    residual_proj = torch.matmul(residual, self.residual_matrix[i])
                    x = x + residual_proj

                # Update residual for next layer
                residual = x

        return x, _
