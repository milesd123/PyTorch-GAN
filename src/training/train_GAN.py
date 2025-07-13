import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ..model.discriminator import Discriminator
from ..model.generator import Generator
from ..model.vgg import VGGLoss
from model_outputs.checkpoints import save_checkpoint
from pair_image import PairedImageDataset

# configure for training
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sd_dir = "SD"
hd_dir = "HD"
batch_size = 8
epochs = 10
# Only update discriminator every N steps
d_update_interval = 3

# load models
G = Generator().to(device)
D = Discriminator().to(device)
vgg_loss = VGGLoss().to(device)

# optimizers
opt_G = optim.Adam(G.parameters(), lr=1e-4)
opt_D = optim.Adam(D.parameters(), lr=1e-4)

# dataset
dataset = PairedImageDataset(sd_dir, hd_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# loss functions
adv_criterion = nn.BCELoss()
pixel_loss = nn.L1Loss()

# Training loop
for epoch in range(epochs):
        for i, (lr_imgs, hr_imgs, filenames) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            valid = torch.ones(hr_imgs.size(0), 1).to(device)
            fake = torch.zeros(hr_imgs.size(0), 1).to(device)

            # Only train discriminator every so many generator steps
            # Helps the generator quality catch up in training
            if i % d_update_interval == 0:
                with torch.no_grad():
                    gen_imgs = G(lr_imgs)

                real_loss = adv_criterion(D(hr_imgs), valid)
                fake_loss = adv_criterion(D(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()
            else:
                d_loss = torch.tensor(0.0, device=device)  # dummy loss for logging

            # Generator forward
            gen_imgs = G(lr_imgs)
            adv_loss = adv_criterion(D(gen_imgs), valid)
            vgg = vgg_loss(gen_imgs, hr_imgs)
            px = pixel_loss(gen_imgs, hr_imgs)

            # Generator total loss
            g_loss = 1e-3 * adv_loss + 1e-2 * vgg + px

            # Generator backpropogation
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            print(f"[{epoch}/{i}] File: {filenames[0]} | G: {g_loss.item():.4f} | D: {d_loss.item():.4f}")

        save_checkpoint(G, opt_G, "model_outputs/G.pth")
        save_checkpoint(D, opt_D, "model_outputs/D.pth")