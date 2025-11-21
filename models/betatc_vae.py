import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import math
from typing import List, Callable, Union, Any, TypeVar, Tuple

from utils.objectives import mse_recon_batch_mean, mse_recon_mean, bce_recon_batch_mean, bce_recon_mean, laplacian_recon_batch_mean, laplacian_recon_mean
Tensor = TypeVar('torch.tensor')


class BetaTCVAE(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        anneal_steps: int = 200,
        alpha: float = 1.0,
        beta: float = 6.0,
        gamma: float = 1.0,
        input_size: int = 32,
        dataset_size: int = None,
        output_activation: str = "tanh",
        recons_dist: str = "gaussian",
        **kwargs
    ) -> None:
        super(BetaTCVAE, self).__init__()

        recon_obj = None
        if recons_dist == "gaussian":
            recon_obj = mse_recon_mean
            if output_activation == "tanh":
                pass  # Keep tanh
            else:
                output_activation = "tanh"  # Default to tanh for gaussian
        elif recons_dist == "bernoulli":
            recon_obj = bce_recon_mean
            output_activation = "sigmoid"
        elif recons_dist == "laplacian":
            recon_obj = laplacian_recon_mean
            if output_activation == "tanh":
                pass  # Keep tanh
            else:
                output_activation = "tanh"  # Default to tanh for laplacian
        else:
            raise ValueError(f"Reconstruction distribution {recons_dist} not supported. Choose from: gaussian, bernoulli, laplacian")

        self.latent_dim = latent_dim
        self.anneal_steps = anneal_steps
        self.input_size = input_size
        self.in_channels = in_channels
        self.dataset_size = dataset_size  # Used for M_N calculation

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Set up objectives dict for compatibility with training loop
        # These keys match the loss function return keys
        self.objectives = {
            "reconstruction_loss": recon_obj,
            "mi_loss": None,
            "tc_loss": None,
            "kld": None,
        }

        self.features = ["mu", "log_var"]

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 32, 32, 32]

        self.hidden_dims = hidden_dims

        # Setup output activation
        self.output_activation = None
        if output_activation == "tanh":
            self.output_activation = nn.Tanh
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid
        elif output_activation == "none":
            self.output_activation = nn.Identity
        else:
            raise ValueError(f"Output activation {output_activation} not supported")

        # Build Encoder
        encoder_in_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        encoder_in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            encoder_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Calculate spatial dimensions after encoder
        num_layers = len(hidden_dims)
        spatial_dim = input_size // (2 ** num_layers)
        encoder_output_size = hidden_dims[-1] * spatial_dim * spatial_dim
        self.encoder_output_size = encoder_output_size

        self.fc = nn.Linear(encoder_output_size, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        # Build Decoder
        modules = []

        # Calculate decoder input size to match encoder output spatial dimensions
        self.decoder_input = nn.Linear(latent_dim, encoder_output_size)

        hidden_dims_reversed = hidden_dims.copy()
        hidden_dims_reversed.reverse()

        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims_reversed[i],
                        hidden_dims_reversed[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_reversed[-1],
                hidden_dims_reversed[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims_reversed[-1], out_channels=in_channels, kernel_size=3, padding=1
            ),
            self.output_activation(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        result = torch.flatten(result, start_dim=1)
        result = self.fc(result)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # Reshape to match the first decoder layer input
        # The first hidden dim in reversed order is hidden_dims[-1]
        num_layers = len(self.hidden_dims)
        spatial_dim = self.input_size // (2 ** num_layers)
        first_hidden_dim = self.hidden_dims[-1]
        result = result.view(-1, first_hidden_dim, spatial_dim, spatial_dim)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> dict:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return {"recons": recons, "input": input, "mu": mu, "log_var": log_var, "z": z}

    def log_density_gaussian(self, x: Tensor, mu: Tensor, logvar: Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at which Gaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss_function(self, inputs: Tensor, args: dict) -> dict:
        """
        Computes the BetaTC-VAE loss function.
        :param inputs: Input images
        :param args: Dictionary containing model outputs (recons, mu, log_var, z)
        :return: Dictionary of losses
        """
        recons = args["recons"]
        mu = args["mu"]
        log_var = args["log_var"]
        z = args["z"]

        batch_size, latent_dim = z.shape

        # Calculate M_N from batch size and dataset size
        if self.dataset_size is not None:
            M_N = batch_size / self.dataset_size
        else:
            # Default: assume batch_size / dataset_size ratio
            # This is a reasonable default for most datasets
            M_N = batch_size / 50000  # Default dataset size estimate

        weight = 1  # Account for the minibatch samples from the dataset

        recons_loss = self.objectives["reconstruction_loss"](inputs, recons)

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

        mat_log_q_z = self.log_density_gaussian(
            z.view(batch_size, 1, latent_dim),
            mu.view(1, batch_size, latent_dim),
            log_var.view(1, batch_size, latent_dim),
        )

        # Reference
        # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
        dataset_size = (1 / M_N) * batch_size  # dataset size
        strat_weight = (dataset_size - batch_size + 1) / (
            dataset_size * (batch_size - 1)
        )
        importance_weights = (
            torch.Tensor(batch_size, batch_size)
            .fill_(1 / (batch_size - 1))
            .to(inputs.device)
        )
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.0

        # Note: The total loss is computed as sum of all returned losses in training loop
        # So we return the individual components scaled appropriately
        return {
            "reconstruction_loss": recons_loss,
            "mi_loss": self.alpha * mi_loss,
            "tc_loss": weight * self.beta * tc_loss,
            "kld": weight * anneal_rate * self.gamma * kld_loss,
        }

    def sample(self, num_samples: int, device: int = None, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param device: (Int/str) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)["recons"]

    def total_trainable_params(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_summary(self):
        """Print model summary using torchsummary."""
        return summary(self, (self.in_channels, self.input_size, self.input_size))
