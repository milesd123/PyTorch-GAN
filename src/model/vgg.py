from torchvision.models import vgg19
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Using vgg19
        vgg = vgg19(pretrained=True).features.eval()
        for param in vgg.parameters(): # Dont update params (wasting compute)
            param.requires_grad = False
        self.vgg = create_feature_extractor(vgg, return_nodes={'35': 'features'})  # relu5_4
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)['features']
        y_vgg = self.vgg(y)['features']
        return self.criterion(x_vgg, y_vgg)
