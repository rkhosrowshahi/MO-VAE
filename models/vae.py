from math import ceil, floor

import torch
import torch.nn as nn
from torchsummary import summary

from utils.objectives import mse_recon_sum, mse_recon_mean, bce_recon_sum, bce_recon_mean, kl_divergence_mean, kl_divergence_sum


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
class VAE(nn.Module):
    def __init__(self, latent_dim=2, input_size=32, in_channels=3, hidden_dims=None, layer_norm="batch", output_activation="tanh", objs=["mse_sum", "kl_sum"], beta=1.0):
        super(VAE, self).__init__()
        
        recon_obj = None
        kl_obj = None
        if "mse_sum" in objs:
            recon_obj = mse_recon_sum
        elif "mse_mean" in objs:
            recon_obj = mse_recon_mean
        elif "bce_sum" in objs:
            recon_obj = bce_recon_sum
            output_activation = "sigmoid"
        elif "bce_mean" in objs:
            recon_obj = bce_recon_mean
            output_activation = "sigmoid"
        else:
            raise ValueError(f"Reconstruction objective {objs} not supported")
        if "kl_mean" in objs:
            kl_obj = kl_divergence_mean
        elif "kl_sum" in objs:
            kl_obj = kl_divergence_sum
        else:
            raise ValueError(f"KL divergence objective {objs} not supported")

        self.objectives = {"reconstruction_loss": recon_obj, "kl_loss": kl_obj}
        
        self.recon_obj = recon_obj
        self.kl_obj = kl_obj

        self.beta = beta
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        # Calculate spatial dimensions after len(hidden_dims) stride-2 convolutions
        # With stride=2, padding=1, kernel=3: each layer halves the spatial dimension
        # After n convolutions: input_size -> input_size//(2^n)
        self.input_size = input_size
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        num_layers = len(hidden_dims)
        spatial_dim = input_size // (2 ** num_layers)
        encoder_output_size = hidden_dims[-1] * spatial_dim * spatial_dim

        if layer_norm == "batch":
            layer_norm = nn.BatchNorm2d
        elif layer_norm == "layer":
            layer_norm = nn.LayerNorm
        elif layer_norm == "none":
            layer_norm = nn.Identity
        else:
            raise ValueError(f"Layer norm {layer_norm} not supported")

        if output_activation == "tanh":
            output_activation = nn.Tanh
        elif output_activation == "sigmoid":
            output_activation = nn.Sigmoid
        elif output_activation == "none":
            output_activation = nn.Identity
        else:
            raise ValueError(f"Output activation {output_activation} not supported")
        
        # Build Encoder
        modules = []
        encoder_in_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(encoder_in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    layer_norm(h_dim),
                    nn.LeakyReLU())
            )
            encoder_in_channels = h_dim
        
        modules.append(nn.Flatten())  # Flatten before linear layers
        self.encoder = nn.Sequential(*modules)
        
        # Latent space
        self.mu = nn.Linear(encoder_output_size, latent_dim)  # mean of the latent space
        self.log_var = nn.Linear(encoder_output_size, latent_dim)  # log variance of the latent space
        
        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, encoder_output_size)
        
        modules = []
        hidden_dims_reversed = hidden_dims.copy()
        hidden_dims_reversed.reverse()
        
        # Add Unflatten layer
        modules.append(nn.Unflatten(1, (hidden_dims[-1], spatial_dim, spatial_dim)))
        
        # Build decoder layers (reverse of encoder)
        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_reversed[i],
                                       hidden_dims_reversed[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    layer_norm(hidden_dims_reversed[i + 1]),
                    nn.LeakyReLU())
            )
        
        # Final layer: last transpose conv + final conv + activation
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims_reversed[-1],
                               hidden_dims_reversed[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            layer_norm(hidden_dims_reversed[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims_reversed[-1], out_channels=in_channels,
                      kernel_size=3, padding=1),
            output_activation()
        )
        
        self.decoder = nn.Sequential(*modules)

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
        out = self.decoder_input(z)
        out = self.decoder(out)
        recons = self.final_layer(out)
        return recons

    def forward(self, x):
        # Forward pass
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return {"recons": recons, "mu": mu, "log_var": log_var, "z": z}

    def total_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def loss_function(self, inputs, args: dict) -> dict:
        recons = args["recons"]
        mu = args["mu"]
        log_var = args["log_var"]

        recon_loss = self.objectives["reconstruction_loss"](inputs, recons)
        kl_loss = self.beta * self.objectives["kl_loss"](mu, log_var)
        

        return {"reconstruction_loss": recon_loss, "kl_loss": kl_loss}

    def sample(self, num_samples=1, device=None):
        """
        Sample from the latent space and generate images.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated images
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            generated_samples = self.decode(z)
        return generated_samples

    def print_model_summary(self):
        return summary(self, (self.in_channels, self.input_size, self.input_size))