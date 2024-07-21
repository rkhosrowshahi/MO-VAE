from math import ceil, floor

import torch
import torch.nn as nn


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class UnFlatten(nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()
    def forward(self, input, size=128):
        return input.view(-1, size, 4, 4)


# Define the VAE model
class VAE3(nn.Module):
    def __init__(self, latent_dim=2, in_height=32, in_channels=3):
        super(VAE3, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 32x32x3 -> 16x16x32
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # PrintLayer(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16x32 -> 8x8x64
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # PrintLayer(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 8x8x64 -> 4x4x128
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            # PrintLayer(),
            nn.Flatten(),  # 26x26x128 -> 128x26x26
            # PrintLayer(),
            # nn.Linear(128 * int((in_height/8)) * int(floor(in_height/8)), 16)
        )
        # Latent space
        self.mu = nn.Linear(128 * int((in_height/8)) * int(floor(in_height/8)), latent_dim)  # mean of the latent space
        self.log_var = nn.Linear(128 * int((in_height/8)) * int(floor(in_height/8)), latent_dim)  # log variance of the latent space
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * int(floor(in_height/8)) * int(floor(in_height/8))),  # 128 -> 128x4x4
            # UnFlatten(),  # 128x4x4 -> 4x4x128
            nn.Unflatten(1, (128, int(floor(in_height/8)), int(floor(in_height/8)))),
            # PrintLayer(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4x128 -> 8x8x64
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # PrintLayer(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8x64 ->   16x16x32
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x32 -> 32x32x3
            # PrintLayer(),
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            # PrintLayer(),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def encode(self, x):
        # Encode the input image
        h = self.encoder(x)
        mu, log_var = self.mu(h), self.log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Reparameterize the latent space
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        # Decode the latent code
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def forward(self, x):
        # Forward pass
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var, z
