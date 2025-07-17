import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.model.discriminator import Discriminator
from src.model.generator import Generator
from src.training.helpers.checkpoints import save_checkpoint
from src.training.helpers.checkpoints import load_checkpoint
from src.training.helpers.pair_image import PairedImageDataset
from src.training.helpers.edge_loss import edge_loss
from src.training.helpers.inference import save_inference_alpha
import time

# configure for training
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sd_dir = "PyTorch-GAN/src/dataset/16x_color"
hd_dir = "PyTorch-GAN/src/dataset/64x_color"
inference_image_path = "PyTorch-GAN/src/training/diamond_sword.png"
batch_size = 16
epochs = 20
# Only update discriminator every N steps
d_update_interval = 4

# load models
G = Generator().to(device)
D = Discriminator().to(device)

# optimizers
opt_G = optim.Adam(G.parameters(), lr=1e-4)
opt_D = optim.Adam(D.parameters(), lr=1e-4)

# Load model weights from generator if it exists
try:
    load_checkpoint(G, opt_G, "models/gen_1.pth", device)
    load_checkpoint(D, opt_D, "models/dis_1.pth", device)
    print("Loaded pretrained Generator weights.")
except Exception as e:
    print(f"Failed to load pretrained generator: {e}")

# dataset
dataset = PairedImageDataset(sd_dir, hd_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# loss functions
adv_criterion = nn.BCELoss()
l1 = nn.L1Loss()

# -------------- Training --------------
total_batches = len(dataloader)
start = time.time()
for epoch in range(epochs):
    for i, (lr_imgs, hr_imgs, filenames) in enumerate(dataloader):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        lr_alpha = lr_imgs[:, 3:4, :, :]
        hr_alpha = hr_imgs[:, 3:4, :, :]

        valid = torch.ones(hr_alpha.size(0), 1).to(device)
        fake = torch.zeros(hr_alpha.size(0), 1).to(device)

        # ----- Update Discriminator -----
        if i % d_update_interval == 0:
            with torch.no_grad():
                gen_alpha = G(lr_alpha)

            real_loss = adv_criterion(D(hr_alpha), valid)
            fake_loss = adv_criterion(D(gen_alpha.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
        else:
            d_loss = torch.tensor(0.0, device=device)

        # ----- Update Generator -----
        gen_alpha = G(lr_alpha)

        alpha_clamped =  gen_alpha.clamp(1e-6,1 - 1e-6)

        loss_entropy = (alpha_clamped * (1 - alpha_clamped)).mean()
        adv_loss = adv_criterion(D(gen_alpha), valid)
        l1_loss = l1(gen_alpha, hr_alpha)
        edge = edge_loss(gen_alpha, hr_alpha)

        # Final generator loss
        g_loss = (
                1e-3 * adv_loss
                  + 1.0 * l1_loss
                  + 1.0 * edge
                  + .5 * loss_entropy
                  )

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        print(f"{(time.time() - start )/60:.2f}m Epoch {epoch+1}/{epochs} Batch {i+1}/{total_batches} File {filenames[0]} | G: {g_loss.item():.4f} D: {d_loss.item():.4f}")

    save_inference_alpha(G, inference_image_path, epoch)
    save_checkpoint(G, opt_G, "models/gen_2.pth")
    save_checkpoint(D, opt_D, "models/dis_2.pth")

print("Done training.")