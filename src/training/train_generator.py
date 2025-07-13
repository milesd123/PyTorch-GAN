import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ..model.generator import Generator
from ..model.vgg import VGGLoss
from model_outputs.checkpoints import save_checkpoint
from pair_image import PairedImageDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Training Parameters
batch_size = 8
epochs = 5

# Dataset Directory
hd_dir = ""
sd_dir = ""

# Initialize Models and optimizers
generator = Generator().to(device)
vgg_loss = VGGLoss().to(device)
pixel_loss = nn.L1Loss()
opt = optim.Adam(generator.parameters(), lr=1e-4)

# Load data
dataset = PairedImageDataset(sd_dir, hd_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Train!!!
for epoch in range(epochs):
    for i, (sd, hd, filename) in enumerate(dataloader):
        sd = sd.to(device)
        hd = hd.to(device)

        gen = generator(sd)
        loss_pixel = pixel_loss(gen, hd)
        loss_vgg = vgg_loss(gen, hd)
        loss = loss_pixel + 1e-2 * loss_vgg

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"[{epoch}/{i}] File: {filename[0]} — Pixel: {loss_pixel.item():.4f}, VGG: {loss_vgg.item():.4f}")

    save_checkpoint(generator, opt, "g_only.pth")
