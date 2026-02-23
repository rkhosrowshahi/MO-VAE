"""
Recursive KL VAE: KL divergence is computed on the encoding of the reconstruction.

Standard VAE: KLD is on z ~ q(z|x).
Recursive KL VAE: KLD is on hat_z ~ q(z|hat_x) where hat_x = dec(enc(x)), i.e.
  hat_z = enc(dec(enc(x))) — second encoder pass on the reconstruction.

Two returned losses:
- reconstruction_loss = lambda_recon * reconstruction_term
- recursive_kld_loss  = anneal_rate * lambda_KL * recursive_KL(q(z|hat_x) || p(z))

KL Annealing
------------
  recursive_kld is adversarial early in training: when recons is poor,
  enc(recons) gives a large arbitrary μ̂ that injects a mode-averaging gradient
  into the decoder before it has learned basic reconstruction.

  Annealing ramps the recursive_kld weight linearly from 0 → λ_KL over the
  first `recursive_kld_anneal_steps` gradient steps:

      anneal_rate(t) = min(t / anneal_steps, 1.0)

  Config parameter: recursive_kld_anneal_steps (default: 25000)
"""

from .vae import VAE


class RecursiveKLVAE(VAE):
    """
    VAE that uses a second encoder pass for the KL term (recursive KL),
    with linear annealing to prevent adversarial early-training gradients.

    Forward:  x -> enc(x) -> z -> dec(z) -> recons
    KL pass:  recons -> enc(recons) -> (mu_hat, log_var_hat); KLD(mu_hat, log_var_hat)

    loss_weights: [lambda_recon, lambda_recursive_kld]
    recursive_kld_anneal_steps: int  (default 25000)
    """

    num_iter = 0  # global step counter, same convention as BetaTCVAE

    def __init__(self, recursive_kld_anneal_steps: int = 25000, **kwargs):
        # Base VAE expects 2 lambda_weights [recon, kld]; we accept 2 or 3 [recon, kld, recursive_kld]
        lambda_weights = kwargs.get("lambda_weights", [1.0, 0.00025])
        if isinstance(lambda_weights, list) and len(lambda_weights) >= 2:
            kwargs = {**kwargs, "lambda_weights": lambda_weights[:2]}
        super().__init__(**kwargs)

        self.anneal_steps = recursive_kld_anneal_steps

        # No task-specific heads: all params are shared. Use backward() not mtl_backward()
        # so we don't require task_params vs shared_params split (avoids torchjd error).
        self.features = None
        # Rename kld_loss → recursive_kld_loss
        self.objectives["recursive_kld_loss"] = self.objectives.pop("kld_loss")
        kld_weight = self.lambda_weights.pop("kld_loss")
        self.lambda_weights["recursive_kld_loss"] = (
            lambda_weights[2] if isinstance(lambda_weights, list) and len(lambda_weights) >= 3
            else kld_weight
        )

    def forward(self, x):
        # First pass: encode -> decode (reconstruction)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        # Second pass: encode reconstruction for KL
        mu_hat, log_var_hat = self.encode(recons)
        return {
            "recons": recons,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "mu_hat": mu_hat,
            "log_var_hat": log_var_hat,
        }

    def loss_function(self, inputs, args: dict) -> dict:
        recons = args["recons"]
        mu_hat, log_var_hat = args["mu_hat"], args["log_var_hat"]

        recon_loss = self.objectives["reconstruction_loss"](inputs, recons)
        recursive_kld = self.objectives["recursive_kld_loss"](mu_hat, log_var_hat)

        lambda_recon = self.lambda_weights["reconstruction_loss"]
        lambda_recursive_kld = self.lambda_weights["recursive_kld_loss"]

        # Linear annealing: 0 → λ_recursive_kld over anneal_steps.
        if self.training:
            RecursiveKLVAE.num_iter += 1
            anneal_rate = min(RecursiveKLVAE.num_iter / self.anneal_steps, 1.0)
        else:
            anneal_rate = 1.0

        weighted_recon_loss = lambda_recon * recon_loss
        weighted_recursive_kld = anneal_rate * lambda_recursive_kld * recursive_kld
        total_loss = weighted_recon_loss + weighted_recursive_kld

        return {
            "reconstruction_loss": weighted_recon_loss,
            "recursive_kld_loss": weighted_recursive_kld,
            "total_loss": total_loss,
        }
