import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    def __init__(self, sd_dir, hd_dir):
        self.sd_paths = sorted([
            os.path.join(sd_dir, f) for f in os.listdir(sd_dir) if f.endswith((".jpg", ".png"))
        ])
        self.hd_paths = sorted([
            os.path.join(hd_dir, f) for f in os.listdir(hd_dir) if f.endswith((".jpg", ".png"))
        ])
        assert len(self.sd_paths) == len(self.hd_paths), "SD and HD folder must have same number of images"

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.sd_paths)

    def __getitem__(self, idx):
        sd_path = self.sd_paths[idx]
        hd_path = self.hd_paths[idx]

        #Convert to rgba if it is not already
        sd_img_pil = Image.open(sd_path).convert("RGBA")
        hd_img_pil = Image.open(hd_path).convert("RGBA")

        #Make sure image sizes are correct
        if sd_img_pil.size != (16, 16):
            print(f"Bad SD: {sd_path} size={sd_img_pil.size}")
        if hd_img_pil.size != (64, 64):
            print(f"Bad HD: {hd_path} size={hd_img_pil.size}")

        sd_img = self.to_tensor(sd_img_pil)
        hd_img = self.to_tensor(hd_img_pil)

        filename = os.path.basename(sd_path)
        return sd_img, hd_img, filename # Return as tensors
