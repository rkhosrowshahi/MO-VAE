import torch
import torch.nn as nn
from .types_ import *


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
    def __init__(
        self,
        latent_dim=2,
        in_size=32,
        in_channels=3,
        hidden_dims=[32, 64, 128, 256, 512],
    ):
        super(VAE3, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_size = in_size
        self.final_dim = hidden_dims[-1]
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, in_size, in_size))
        self.size = out.shape[2]
        # Latent Space
        self.mu = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)
        # Decoder
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(
            latent_dim, hidden_dims[-1] * self.size * self.size
        )
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        modules.append(self.final_layer)
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        # Encode the input image
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
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
        out = self.decoder_input(z)
        out = out.view(-1, self.final_dim, self.size, self.size)
        x_reconstructed = self.decoder(out)
        return x_reconstructed

    def forward(self, x):
        # Forward pass
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var, z

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        # Forward pass
        self.forward(x)[0]

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
