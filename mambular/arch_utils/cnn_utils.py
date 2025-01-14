import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        conv_output = self.conv(x)  # Standard convolution
        attention_map = torch.sigmoid(self.attention(x))  # Generate attention map
        return conv_output * attention_map  # Apply attention


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.se_fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.se_fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, x):
        batch, channels, height, width = x.size()

        # Channel attention
        avg_pool = x.view(batch, channels, -1).mean(dim=2)  # (B, C)
        max_pool, _ = x.view(batch, channels, -1).max(dim=2)  # (B, C)
        channel_weights = F.sigmoid(
            self.se_fc2(F.relu(self.se_fc1(avg_pool)))
            + self.se_fc2(F.relu(self.se_fc1(max_pool)))
        ).view(batch, channels, 1, 1)
        x = x * channel_weights

        # Spatial attention
        avg_pool = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = x.max(dim=1, keepdim=True)  # (B, 1, H, W)
        spatial_attention = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1))
        )
        return x * spatial_attention


class CNNBlock(nn.Module):
    """
    A modular CNN block that allows for configurable convolutional, pooling, and dropout layers.

    Attributes
    ----------
    cnn : nn.Sequential
        A sequential container holding the convolutional, activation, pooling, and dropout layers.

    Methods
    -------
    forward(x):
        Defines the forward pass of the CNNBlock.
    """

    def __init__(self, config):
        super().__init__()
        layers = []
        in_channels = config.input_channels
        self.attention = config.attention

        # Ensure dropout_positions is a list
        dropout_positions = config.dropout_positions or []

        for i in range(config.num_layers):
            # Convolutional layer with different attention mechanisms
            if self.attention == "attention":
                layers.append(
                    SelfAttentionConv2D(
                        in_channels=in_channels,
                        out_channels=config.out_channels_list[i],
                        kernel_size=config.kernel_size_list[i],
                    )
                )
            elif self.attention == "cbam":
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=config.out_channels_list[i],
                        kernel_size=config.kernel_size_list[i],
                        stride=config.stride_list[i],
                        padding=config.padding_list[i],
                    )
                )
                layers.append(CBAMBlock(in_channels=config.out_channels_list[i]))
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=config.out_channels_list[i],
                        kernel_size=config.kernel_size_list[i],
                        stride=config.stride_list[i],
                        padding=config.padding_list[i],
                    )
                )
            layers.append(nn.ReLU())

            # Pooling layer
            if config.pooling_method == "max":
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=config.pooling_kernel_size_list[i],
                        stride=config.pooling_stride_list[i],
                    )
                )
            elif config.pooling_method == "avg":
                layers.append(
                    nn.AvgPool2d(
                        kernel_size=config.pooling_kernel_size_list[i],
                        stride=config.pooling_stride_list[i],
                    )
                )

            # Dropout layer
            if i in dropout_positions:
                layers.append(nn.Dropout(p=config.dropout_rate))

            in_channels = config.out_channels_list[i]

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input has shape (N, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.cnn(x)
