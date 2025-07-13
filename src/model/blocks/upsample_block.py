import torch.nn as nn

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        # Upsample, this is what makes the image bigger(PixelShuffle)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),        # Reduces channel amount
            nn.PReLU() # Has trainable parameter
        )

    def forward(self, x):
        return self.block(x)