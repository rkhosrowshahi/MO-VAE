"""
Recursive Cyclic VAE (RC-VAE).

Combines two complementary objectives to solve the problems of both CycleVAE
and PSR-VAE:

  CycleVAE problem:  no KL → encoder not regularized toward prior → poor generation
  PSR-VAE problem:   standard KL → sparse decoder Jacobian → biased MTL aggregation

Solution: replace standard KL with recursive KL + add latent cycle consistency.

Three objectives
----------------
  1. reconstruction_loss  =  λ_recon · recon(x, dec(enc(x)))
        Pure reconstruction — no KL folded in.
        Gradient reaches encoder (via reparameterization) and decoder.

  2. recursive_kld_loss  =  λ_rec_kl · KL(enc(dec(enc(x))) ‖ p(z))
        Encoder regularization via the reconstruction path.
        Gradient chain: KL → (μ̂,σ̂) → enc(recons) → recons=dec(z) → θ_dec
                        KL → (μ̂,σ̂) → enc(recons) → θ_enc  (second enc pass)
                        KL → dec(z) → z → enc(x) → θ_enc    (first enc pass)
        → Non-zero decoder Jacobian (unlike standard KL).
        → No sparse Jacobian → no MTL aggregation bias.
        → Converges to standard KL when reconstruction is perfect.

  3. cycle_loss  =  λ_cycle · ‖z_prior − enc_mean(dec(z_prior))‖²
        Generative right-inverse constraint — anti-mode-averaging.
        Gradient: 2(μ_gen − z_prior) — unique target per z_prior sample.
        Counteracts the mode-averaging tendency of recursive KL on the decoder.

Jacobian structure
------------------
  All three objectives have non-zero gradients w.r.t. both encoder AND decoder
  parameters.  No row is sparse → MTL aggregators (UPGrad, MGDA, Aligned-MTL)
  operate without the systematic bias caused by zero-padded KL Jacobians.

  | Objective      | θ_enc   | θ_dec   |
  |----------------|---------|---------|
  | recon          | non-zero| non-zero|
  | recursive_kld  | non-zero| non-zero|  ← key difference vs standard KL
  | cycle          | non-zero| non-zero|

  The three-way tension is genuine and meaningful:
  - recon  ↔  recursive_kld:  reconstruction quality vs latent regularization
  - recon  ↔  cycle:          real-data decoding vs prior-sample decoding
  - recursive_kld  ↔  cycle:  mode-averaging pressure vs anti-averaging pressure

Loss weights
------------
  loss_weights: [lambda_recon, lambda_recursive_kld, lambda_cycle]
  Recommended defaults: [1.0, 0.00025, 0.00025]
    - Both regularization terms start equal; tune lambda_cycle up if generated
      images lack diversity, or tune lambda_recursive_kld down if reconstructions
      remain blurry.
    - Recursive KL can have noisy gradients early in training (when reconstruction
      is poor), so keep lambda_recursive_kld ≤ lambda_recon × 1e-3 initially.

Forward pass
------------
  Branch A (recon + recursive KL):
      x → enc(x) → (μ, σ) → z → dec(z) → recons → enc(recons) → (μ̂, σ̂)

  Branch B (cycle):
      z_prior ~ N(0,I) → dec(z_prior) → x_gen → enc(x_gen) → μ_gen
"""

import torch

from .vae import VAE
from utils.objectives import kl_divergence


def _cycle_loss(z_prior: torch.Tensor, mu_gen: torch.Tensor) -> torch.Tensor:
    """L2 latent cycle loss: mean over batch of sum-of-squares over latent dims."""
    return ((z_prior - mu_gen) ** 2).sum(dim=1).mean()


class RecursiveCyclicVAE(VAE):
    """
    Recursive Cyclic VAE: three-objective model combining reconstruction,
    recursive KL regularization, and latent cycle consistency.

    loss_weights: [lambda_recon, lambda_recursive_kld, lambda_cycle]
    """

    def __init__(self, **kwargs):
        lambda_weights = kwargs.get("lambda_weights", [1.0, 0.00025, 0.00025])
        if isinstance(lambda_weights, list) and len(lambda_weights) >= 3:
            # Pass [recon, 0] to base VAE — we own all KL bookkeeping here.
            kwargs = {**kwargs, "lambda_weights": [lambda_weights[0], 0.0]}
        super().__init__(**kwargs)

        # All three objectives share encoder+decoder — no clean task-specific
        # head split.  Use backward() (not mtl_backward()) so Jacobians are
        # computed w.r.t. all parameters without sparsity bias.
        self.features = None

        self.objectives["recursive_kld_loss"] = kl_divergence
        self.objectives["cycle_loss"] = _cycle_loss

        if isinstance(lambda_weights, list) and len(lambda_weights) >= 3:
            self.lambda_weights["recursive_kld_loss"] = lambda_weights[1]
            self.lambda_weights["cycle_loss"] = lambda_weights[2]
        else:
            self.lambda_weights["recursive_kld_loss"] = 0.00025
            self.lambda_weights["cycle_loss"] = 0.00025

    def forward(self, x):
        # ── Branch A: reconstruction + recursive KL ───────────────────────
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        # Second encoder pass on the reconstruction — provides non-sparse
        # decoder Jacobian for the recursive KL objective.
        mu_hat, log_var_hat = self.encode(recons)

        # ── Branch B: latent cycle consistency ────────────────────────────
        # Fresh z_prior drawn from N(0,I) — independent of Branch A.
        z_prior = torch.randn(x.size(0), self.latent_dim, device=x.device)
        x_gen = self.decode(z_prior)
        mu_gen, log_var_gen = self.encode(x_gen)

        return {
            "recons": recons,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            # Recursive KL branch
            "mu_hat": mu_hat,
            "log_var_hat": log_var_hat,
            # Cycle branch
            "z_prior": z_prior,
            "x_gen": x_gen,
            "mu_gen": mu_gen,
            "log_var_gen": log_var_gen,
        }

    def loss_function(self, inputs, args: dict) -> dict:
        recons = args["recons"]
        mu_hat, log_var_hat = args["mu_hat"], args["log_var_hat"]
        mu_gen = args["mu_gen"]
        z_prior = args["z_prior"]

        # ── Objective 1: pure reconstruction ─────────────────────────────
        recon_loss = self.objectives["reconstruction_loss"](inputs, recons)

        # ── Objective 2: recursive KL — encoder regularization ───────────
        # KL of the re-encoded reconstruction toward the prior.
        # Non-zero decoder gradient: KL → enc(recons) → recons=dec(z) → θ_dec
        recursive_kld = self.objectives["recursive_kld_loss"](mu_hat, log_var_hat)

        # ── Objective 3: latent cycle consistency — generative constraint ─
        # ||z_prior − enc_mean(dec(z_prior))||²
        # Gradient to decoder: 2(μ_gen − z_prior) — unique per sample.
        cycle = self.objectives["cycle_loss"](z_prior, mu_gen)

        lambda_recon = self.lambda_weights["reconstruction_loss"]
        lambda_rec_kl = self.lambda_weights["recursive_kld_loss"]
        lambda_cycle = self.lambda_weights["cycle_loss"]

        weighted_recon_loss = lambda_recon * recon_loss
        weighted_recursive_kld = lambda_rec_kl * recursive_kld
        weighted_cycle_loss = lambda_cycle * cycle

        total_loss = weighted_recon_loss + weighted_recursive_kld + weighted_cycle_loss

        return {
            "reconstruction_loss": weighted_recon_loss,
            "recursive_kld_loss": weighted_recursive_kld,
            "cycle_loss": weighted_cycle_loss,
            "total_loss": total_loss,
        }
