import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time
from src.model.generator import Generator
from src.training.helpers.checkpoints import save_checkpoint
from src.training.helpers.pair_image import PairedImageDataset
from src.training.helpers.inference import save_inference_alpha
from src.training.helpers.edge_loss import edge_loss

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Training Parameters
batch_size = 32
epochs = 15

# Dataset Directory
sd_dir = "/PyTorch-GAN/src/dataset/16x_color"
hd_dir = "/PyTorch-GAN/src/dataset/64x_color"
inference_image_path = "PyTorch-GAN/src/training/diamond_sword.png"

# Initialize Models and optimizers
generator = Generator().to(device)
loss_fn = nn.L1Loss()
bce_loss = nn.BCELoss()
opt = optim.Adam(generator.parameters(), lr=2e-4)

# Dataset / Dataloader
dataset = PairedImageDataset(sd_dir, hd_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
total_batches = len(dataloader)

# Training loop
start = time.time()
for epoch in range(epochs):
    for i, (sd, hd, filename) in enumerate(dataloader):
        # Extract alpha channel only
        input_alpha = sd[:, 3:].to(device)  # (B, 1, 16, 16)
        target_alpha = (hd[:, 3:] > 0.5).float().to(device)  # (B, 1, 64, 64)

        pred_alpha = generator(input_alpha)  # (B, 1, 64, 64)
        pred_alpha_clamped = pred_alpha.clamp(1e-6, 1 - 1e-6)

        loss_l1 = loss_fn(pred_alpha, target_alpha)
        loss_entropy = (pred_alpha_clamped * (1 - pred_alpha_clamped)).mean()
        loss_bce = bce_loss(pred_alpha_clamped, target_alpha)
        loss_edge = edge_loss(pred_alpha, target_alpha)

        # Try high loss_entropy next time, which is the sharpness/ forces values to 0/1
        # OG/best after 5 epochs:
        # loss = loss_l1 + 0.5 * loss_entropy + 1.0 * loss_bce

        loss = 1 * loss_l1 + 2.5 * loss_entropy + 1 * loss_bce + 10 * loss_edge

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"Time: {(time.time() - start) / 60:.4f}m | Epoch: {epoch + 1}/{epochs} | Batch: {i + 1}/{total_batches}")

    save_inference_alpha(generator, inference_image_path, epoch)
    save_checkpoint(generator, opt, "models/generator_alpha_only.pth")

print(f"Training time: {(time.time() - start) / 60:.2f} minutes. Saved to models/generator_alpha_only.pth")