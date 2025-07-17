import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Block: Convolution, batchnorm, leaky Relu
        def block(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        #8 Blocks
        self.model = nn.Sequential(
            block(1, 32, 2), # h/2 = 32
            block(32, 64, 2), # 32/2 = 16

            # Flatten for fully connected layers
            nn.Flatten(),
            # Fully connected
            nn.Linear(64 * 16 * 16, 128), #
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1), # We only want 1,0 so fully connected to 1 feature

            #sigmoid for 0,1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
