import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    def __init__(self, sd_dir, hd_dir):
        # The dataset images will be corresponding integers like 001.png, 002
        self.sd_paths = sorted([os.path.join(sd_dir, f) for f in os.listdir(sd_dir) if f.endswith((".jpg", ".png"))])
        self.hd_paths = sorted([os.path.join(hd_dir, f) for f in os.listdir(hd_dir) if f.endswith((".jpg", ".png"))])
        assert len(self.sd_paths) == len(self.hd_paths), "SD and HD folder must have same number of images"

        # Convert to 0,1 rather than 0,255
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.sd_paths)

    def __getitem__(self, idx):
        sd_path = self.sd_paths[idx]
        hd_path = self.hd_paths[idx]

        sd_img = self.to_tensor(Image.open(sd_path).convert("RGB"))
        hd_img = self.to_tensor(Image.open(hd_path).convert("RGB"))

        filename = os.path.basename(sd_path)
        return sd_img, hd_img, filename