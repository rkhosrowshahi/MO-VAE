"""
Cycle VAE: VAE with latent cycle consistency; reconstruction loss is recon-only (no KLD).

Same two-branch setup as PSR-VAE:
  - Branch 1: reconstruction  x → enc → z → dec → recons
  - Branch 2: cycle          z_prior → dec → x_gen → enc → μ_gen,  loss = ||z_prior - μ_gen||²

Difference from PSR-VAE: the reported and weighted reconstruction_loss is only
λ_recon * recon_loss (no KLD). KLD is still included in total_loss for training.
"""

import torch

from .vae import VAE


def _cycle_loss(z_prior: torch.Tensor, mu_gen: torch.Tensor) -> torch.Tensor:
    """L2 latent cycle loss: mean over batch of sum-of-squares over latent dims."""
    return ((z_prior - mu_gen) ** 2).sum(dim=1).mean()


class CycleVAE(VAE):
    """
    VAE with latent cycle consistency; reconstruction_loss is recon-only (no KLD in that term).
    """

    def __init__(self, **kwargs):
        # Two weights only: [reconstruction, cycle]. KLD is not used.
        lambda_weights = kwargs.get("lambda_weights", [1.0, 0.00025])
        if isinstance(lambda_weights, list) and len(lambda_weights) >= 2:
            # Base VAE expects [recon, kld]; we pass [recon, 0] since we don't use KLD.
            kwargs = {**kwargs, "lambda_weights": [lambda_weights[0], 0.0]}
        super().__init__(**kwargs)

        self.features = None
        self.objectives["cycle_loss"] = _cycle_loss

        if isinstance(lambda_weights, list) and len(lambda_weights) >= 2:
            self.lambda_weights["cycle_loss"] = lambda_weights[1]
        else:
            self.lambda_weights["cycle_loss"] = 0.00025

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)

        z_prior = torch.randn(x.size(0), self.latent_dim, device=x.device)
        x_gen = self.decode(z_prior)
        mu_gen, log_var_gen = self.encode(x_gen)

        return {
            "recons": recons,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "z_prior": z_prior,
            "x_gen": x_gen,
            "mu_gen": mu_gen,
            "log_var_gen": log_var_gen,
        }

    def loss_function(self, inputs, args: dict) -> dict:
        recons = args["recons"]
        mu, log_var = args["mu"], args["log_var"]
        mu_gen = args["mu_gen"]
        z_prior = args["z_prior"]

        recon_loss = self.objectives["reconstruction_loss"](inputs, recons)
        # standard_kld = self.objectives["kld_loss"](mu, log_var)
        cycle = self.objectives["cycle_loss"](z_prior, mu_gen)

        lambda_recon = self.lambda_weights["reconstruction_loss"]
        # lambda_kld = self.lambda_weights["kld_loss"]
        lambda_cycle = self.lambda_weights["cycle_loss"]

        # weighted_kld = lambda_kld * standard_kld
        weighted_recon_loss = lambda_recon * recon_loss
        weighted_cycle_loss = lambda_cycle * cycle

        total_loss = weighted_recon_loss + weighted_cycle_loss

        return {
            "reconstruction_loss": weighted_recon_loss,
            "cycle_loss": weighted_cycle_loss,
            "total_loss": total_loss,
        }
