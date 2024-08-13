import torch
import torch.nn as nn
import torch.nn.functional as F


def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    def forward(self, x):
        return reglu(x)


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        assert x.size(-1) % 2 == 0, "Input dimension must be even"
        split_dim = x.size(-1) // 2
        return x[..., :split_dim] * torch.sigmoid(x[..., split_dim:])


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, activation=F.relu, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(
            *args, activation=activation, **kwargs
        )
        self.custom_activation = activation

        # Check if the activation function is an instance of a GLU variant
        if activation in [ReGLU, GLU] or isinstance(activation, (ReGLU, GLU)):
            self.linear1 = nn.Linear(
                self.linear1.in_features,
                self.linear1.out_features * 2,
                bias=kwargs.get("bias", True),
            )
            self.linear2 = nn.Linear(
                self.linear2.in_features,
                self.linear2.out_features,
                bias=kwargs.get("bias", True),
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Use the provided activation function
        if self.custom_activation in [ReGLU, GLU] or isinstance(
            self.custom_activation, (ReGLU, GLU)
        ):
            src2 = self.linear2(self.custom_activation(self.linear1(src)))
        else:
            src2 = self.linear2(self.custom_activation(self.linear1(src)))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
