"""
Prior Sample Round-Trip VAE (PSR-VAE) with Latent Cycle Consistency.

Motivation
----------
In a standard VAE the KL term D_KL(q(z|x) || p(z)) has zero gradient w.r.t.
decoder parameters, so the decoder receives no direct signal about whether it
can generate realistic images from the prior N(0, I).

Previous attempt (PSR-KL) used KL(enc(dec(z_prior)) || p(z)) as the generative
objective, but its gradient w.r.t. the decoder is:

    ∂KL/∂μ_gen = μ_gen  →  always pushes toward 0, regardless of z_prior.

This is mode-averaging: the optimal decoder strategy to minimise this loss
is to produce the same blurry mean image for every z_prior, because it
consistently encodes near (0, 0).

Fix: Latent Cycle Consistency
------------------------------
Replace the KL with an L2 cycle loss:

    L_cycle = ||z_prior - μ_gen||²,   where μ_gen = enc_mean(dec(z_prior))

Its gradient w.r.t. the decoder is:

    ∂L_cycle/∂μ_gen = 2(μ_gen - z_prior)

The target is NOW z_prior itself — a different vector for every sample.  The
decoder CANNOT collapse to a single average output, because each z_prior
demands a different image to minimise the loss.  The only stable fixed point
is dec ≈ enc⁻¹: i.e. the decoder is a genuine right-inverse of the encoder
on the prior support.

Two objectives
--------------
  1. reconstruction_loss  =  recon(x, dec(enc(x)))  +  λ_kl · KL(q(z|x) ‖ p(z))
        Standard ELBO — trains encoder + decoder on real data.

  2. cycle_loss  =  λ_cycle · ||z_prior - enc_mean(dec(z_prior))||²
        Latent cycle consistency — trains decoder (and encoder) to form a
        genuine inverse pair on prior samples.  No mode averaging.

Because cycle_loss uses a completely separate computational path (fresh
z_prior, independent decode + encode), the two objectives do not share the
same [mu, log_var] features from the first pass.  Therefore self.features =
None and torchjd's backward() (not mtl_backward()) is used, computing
Jacobians w.r.t. all parameters.

Loss weights
------------
  loss_weights: [lambda_recon, lambda_kld, lambda_cycle]
  Recommended defaults: [1.0, 0.00025, 0.00025]
    - lambda_cycle is set equal to lambda_kld as a starting point.
    - At convergence the cycle loss → 0; it is self-regulating.
    - If reconstructions are still weak, try decreasing lambda_cycle (e.g.
      0.0001) to let the reconstruction objective dominate early training.
"""

import torch

from .vae import VAE


def _cycle_loss(z_prior: torch.Tensor, mu_gen: torch.Tensor) -> torch.Tensor:
    """L2 latent cycle loss: mean over batch of sum-of-squares over latent dims."""
    return ((z_prior - mu_gen) ** 2).sum(dim=1).mean()


class PSRVAE(VAE):
    """
    Prior Sample Round-Trip VAE with latent cycle consistency.

    Forward pass has two branches:

    Branch 1 (reconstruction):
        x  →  enc(x)  →  (μ, log_var)  →  z  →  dec(z)  →  recons

    Branch 2 (generative — cycle):
        z_prior ~ N(0, I)  →  dec(z_prior)  →  x_gen
        x_gen  →  enc(x_gen)  →  (μ_gen, log_var_gen)
        loss = ||z_prior - μ_gen||²

    Parameters
    ----------
    All other kwargs are forwarded to VAE.__init__.
    """

    def __init__(self, **kwargs):
        lambda_weights = kwargs.get("lambda_weights", [1.0, 0.00025, 0.00025])
        if isinstance(lambda_weights, list) and len(lambda_weights) >= 3:
            kwargs = {**kwargs, "lambda_weights": lambda_weights[:2]}
        super().__init__(**kwargs)

        # Cycle loss uses a fresh encode/decode path — no shared [mu, log_var]
        # features with the reconstruction branch.  Use backward() (not
        # mtl_backward()) so Jacobians are computed w.r.t. all parameters.
        self.features = None

        self.objectives["cycle_loss"] = _cycle_loss

        if isinstance(lambda_weights, list) and len(lambda_weights) >= 3:
            self.lambda_weights["cycle_loss"] = lambda_weights[2]
        else:
            self.lambda_weights["cycle_loss"] = 0.00025

    def forward(self, x):
        # ── Branch 1: reconstruction ──────────────────────────────────────
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)

        # ── Branch 2: latent cycle consistency ────────────────────────────
        # Draw z_prior independently from N(0, I); same batch size as input.
        z_prior = torch.randn(x.size(0), self.latent_dim, device=x.device)
        x_gen = self.decode(z_prior)

        # Only mu_gen is needed for the cycle loss; log_var_gen is retained
        # in the output dict for monitoring / future use.
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

        # ── Objective 1: ELBO (reconstruction + standard KL) ─────────────
        recon_loss = self.objectives["reconstruction_loss"](inputs, recons)
        standard_kld = self.objectives["kld_loss"](mu, log_var)

        # ── Objective 2: latent cycle consistency ─────────────────────────
        # ||z_prior - enc_mean(dec(z_prior))||²
        # Gradient w.r.t. decoder: 2(μ_gen - z_prior) — unique target per
        # sample, preventing the mode-averaging collapse of the KL formulation.
        cycle = self.objectives["cycle_loss"](z_prior, mu_gen)

        lambda_recon = self.lambda_weights["reconstruction_loss"]
        lambda_kld = self.lambda_weights["kld_loss"]
        lambda_cycle = self.lambda_weights["cycle_loss"]

        # Standard KL folded into reconstruction_loss (ELBO form) so the two
        # reported objectives remain semantically distinct.
        weighted_kld = lambda_kld * standard_kld
        weighted_recon_loss = lambda_recon * recon_loss + weighted_kld
        weighted_cycle_loss = lambda_cycle * cycle

        total_loss = weighted_recon_loss + weighted_cycle_loss

        return {
            "reconstruction_loss": weighted_recon_loss,
            "cycle_loss": weighted_cycle_loss,
            "total_loss": total_loss,
        }
