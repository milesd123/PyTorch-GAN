from torchvision import transforms
from PIL import Image
import os
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def preprocess_alpha(image_path, size=(16, 16)):
    img = Image.open(image_path).convert("RGBA").resize(size, Image.NEAREST)
    alpha = transforms.ToTensor()(img)[3].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return alpha.to(device)

def save_inference_alpha(model, path, epoch):
    model.eval()
    with torch.no_grad():
        input_alpha = preprocess_alpha(path)
        output = model(input_alpha)
        binary = (output > 0.5).float().squeeze(0).squeeze(0)  # (H, W)
        out_img = transforms.ToPILImage()(binary.cpu())
        os.makedirs("inference_outputs_alpha_only", exist_ok=True)
        out_img.save(f"inference_outputs_alpha_only/epoch_{epoch + 1}.png")
    model.train()