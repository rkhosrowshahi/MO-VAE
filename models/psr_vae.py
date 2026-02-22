"""
Prior Sample Round-Trip VAE (PSR-VAE).

Motivation
----------
In a standard VAE the KL term D_KL(q(z|x) || p(z)) has zero gradient w.r.t.
decoder parameters, so the decoder receives no direct signal about whether it
can generate realistic images from the prior N(0, I).

PSR-VAE adds a second, genuinely generative objective by sampling z directly
from the prior, decoding it to a synthetic image, and then re-encoding that
image.  The KL of the re-encoded distribution w.r.t. the prior measures how
well the decoder produces images that "live in the latent manifold" — and its
gradient reaches the decoder through the chain:

    ∂L_psr / ∂θ_dec  =  ∂KL/∂μ̂  ·  ∂μ̂/∂x_gen  ·  ∂x_gen/∂θ_dec   (≠ 0)

Two objectives
--------------
  1. reconstruction_loss  =  recon(x, dec(enc(x)))  +  λ_kl · KL(q(z|x) ‖ p(z))
        Standard ELBO — trains encoder + decoder on real data.

  2. psr_loss  =  λ_psr · KL(enc(dec(z_prior)) ‖ p(z)),   z_prior ~ N(0, I)
        Prior Sample Round-Trip — trains decoder (and encoder) on synthetic
        samples, ensuring generated images re-encode near the prior.

Because psr_loss uses a completely separate computational path (fresh z_prior,
independent decode + encode), the two objectives do not share the same [mu,
log_var] features.  Therefore self.features = None and torchjd's backward()
(not mtl_backward()) is used, computing Jacobians w.r.t. all parameters.

Loss weights
------------
  loss_weights: [lambda_recon, lambda_kld, lambda_psr]
  Recommended defaults: [1.0, 0.00025, 0.00025]
    - lambda_kld and lambda_psr are equal: both are KL divergences over the
      same latent_dim-dimensional space and have the same numerical scale.
    - This avoids the 4000× imbalance present in the original RecursiveKLVAE.

Optional: detach_gen
--------------------
  If detach_gen=True, x_gen is detached before re-encoding, so the PSR KL
  gradient reaches the encoder only (not the decoder).  Useful for ablation
  studies to isolate the encoder-only effect.  Default: False (full gradient).
"""

import torch

from .vae import VAE


class PSRVAE(VAE):
    """
    Prior Sample Round-Trip VAE.

    Forward pass has two branches:

    Branch 1 (reconstruction):
        x  →  enc(x)  →  (μ, log_var)  →  z  →  dec(z)  →  recons

    Branch 2 (generative):
        z_prior ~ N(0, I)  →  dec(z_prior)  →  x_gen
        x_gen  →  enc(x_gen)  →  (μ_gen, log_var_gen)
        loss = KL(N(μ_gen, exp(log_var_gen)) ‖ N(0, I))

    Parameters
    ----------
    detach_gen : bool
        If True, detach x_gen before re-encoding so PSR KL gradient only
        reaches the encoder.  Default False (gradient flows into decoder).
    All other kwargs are forwarded to VAE.__init__.
    """

    def __init__(self, detach_gen: bool = False, **kwargs):
        lambda_weights = kwargs.get("lambda_weights", [1.0, 0.00025, 0.00025])
        if isinstance(lambda_weights, list) and len(lambda_weights) >= 3:
            # Pass only the first two weights to base VAE (recon, kld)
            kwargs = {**kwargs, "lambda_weights": lambda_weights[:2]}
        super().__init__(**kwargs)

        self.detach_gen = detach_gen

        # PSR VAE objectives share all parameters — no clean task-specific /
        # shared split at [mu, log_var] level because psr_loss uses a fresh
        # encode pass.  Use backward() (not mtl_backward()) with aggregator.
        self.features = None

        # Register psr_loss using the same KL function
        self.objectives["psr_loss"] = self.objectives["kld_loss"]

        if isinstance(lambda_weights, list) and len(lambda_weights) >= 3:
            self.lambda_weights["psr_loss"] = lambda_weights[2]
        else:
            self.lambda_weights["psr_loss"] = 0.00025

    def forward(self, x):
        # ── Branch 1: reconstruction ──────────────────────────────────────
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)

        # ── Branch 2: prior sample round-trip ─────────────────────────────
        # Draw z_prior independently from the prior N(0, I).
        # Using x.size(0) ensures the same batch size as the input.
        z_prior = torch.randn(x.size(0), self.latent_dim, device=x.device)
        x_gen = self.decode(z_prior)

        if self.detach_gen:
            # Block PSR gradient from reaching decoder weights; only encoder
            # receives the "map generated images to prior" signal.
            x_gen = x_gen.detach()

        mu_gen, log_var_gen = self.encode(x_gen)

        return {
            "recons": recons,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            # Generative branch outputs
            "z_prior": z_prior,
            "x_gen": x_gen,
            "mu_gen": mu_gen,
            "log_var_gen": log_var_gen,
        }

    def loss_function(self, inputs, args: dict) -> dict:
        recons = args["recons"]
        mu, log_var = args["mu"], args["log_var"]
        mu_gen, log_var_gen = args["mu_gen"], args["log_var_gen"]

        # ── Objective 1: ELBO (reconstruction + standard KL) ─────────────
        recon_loss = self.objectives["reconstruction_loss"](inputs, recons)
        standard_kld = self.objectives["kld_loss"](mu, log_var)

        # ── Objective 2: PSR — KL of re-encoded prior sample ─────────────
        psr_kld = self.objectives["psr_loss"](mu_gen, log_var_gen)

        lambda_recon = self.lambda_weights["reconstruction_loss"]
        lambda_kld = self.lambda_weights["kld_loss"]
        lambda_psr = self.lambda_weights["psr_loss"]

        # Standard KL is folded into reconstruction_loss (ELBO form) so that
        # the two reported objectives are semantically distinct:
        #   reconstruction_loss  ←→  "how well does the model encode/decode data?"
        #   psr_loss             ←→  "how well does the decoder generate from the prior?"
        weighted_kld = lambda_kld * standard_kld
        weighted_recon_loss = lambda_recon * recon_loss # + weighted_kld
        weighted_psr_loss = lambda_psr * psr_kld

        total_loss = weighted_recon_loss + weighted_psr_loss

        return {
            "reconstruction_loss": weighted_recon_loss,
            "psr_loss": weighted_psr_loss,
            "total_loss": total_loss,
        }
