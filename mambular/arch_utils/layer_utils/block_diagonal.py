import torch.nn as nn
import torch


class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks, bias=True):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        assert out_features % num_blocks == 0

        block_out_features = out_features // num_blocks

        self.blocks = nn.ModuleList(
            [
                nn.Linear(in_features, block_out_features, bias=bias)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        x = [block(x) for block in self.blocks]
        x = torch.cat(x, dim=-1)
        return x
