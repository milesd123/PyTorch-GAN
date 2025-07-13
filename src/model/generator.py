import torch
import torch.nn as nn
from blocks.upsample_block import UpsampleBlock
from blocks.residual_block import ResidualBlock

class Generator(nn.Module):
    def __init__(self, n_res_blocks=16):
        super().__init__()

        # semi uniform channel amount
        self.k = 64

        self.entry = nn.Sequential(
            nn.Conv2d(3, self.k, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(self.k) for _ in range(n_res_blocks)])

        self.mid = nn.Sequential(
            nn.Conv2d(self.k, self.k, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.k)
        )

        self.upsample = nn.Sequential(
            UpsampleBlock(self.k, 2),
            UpsampleBlock(self.k, 2),
        )

        self.final = nn.Conv2d(self.k, 3, kernel_size=9, padding=4)
        # Around 1M params

    def forward(self, x):
        # Initial convolution + PReLU
        x1 = self.entry(x)
        # Residual block
        x2 = self.res_blocks(x1)
        # Second solo convolution + BatchNorm
        x3 = self.mid(x2)
        # Upsample block
        x4 = self.upsample(x1 + x3)
        return torch.tanh(self.final(x4)) # final convolution back to 3 output channels
