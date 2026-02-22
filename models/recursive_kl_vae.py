"""
Recursive KL VAE: KL divergence is computed on the encoding of the reconstruction.

Standard VAE: KLD is on z ~ q(z|x).
Recursive KL VAE: KLD is on hat_z ~ q(z|hat_x) where hat_x = dec(enc(x)), i.e.
  hat_z = enc(dec(enc(x))) — second encoder pass on the reconstruction.

Two returned losses:
- reconstruction_loss = reconstruction_term + lambda_KL * standard_KL(q(z|x) || p(z))
  (standard KL is folded in, not a separate loss)
- recursive_kld_loss = lambda_KL * recursive_KL(q(z|hat_x) || p(z))
"""

from .vae import VAE


class RecursiveKLVAE(VAE):
    """
    VAE that uses a second encoder pass for the KL term (recursive KL).

    Forward:  x -> enc(x) -> z -> dec(z) -> recons
    KL pass:  recons -> enc(recons) -> (mu_hat, log_var_hat); KLD(mu_hat, log_var_hat)
    """

    def __init__(self, **kwargs):
        # Base VAE expects 2 lambda_weights [recon, kld]; we use 3 [recon, kld, recursive_kld]
        lambda_weights = kwargs.get("lambda_weights", [1.0, 0.00025, 1.0])
        if isinstance(lambda_weights, list) and len(lambda_weights) >= 3:
            kwargs = {**kwargs, "lambda_weights": lambda_weights[:2]}
        super().__init__(**kwargs)
        # No task-specific heads: all params are shared. Use backward() not mtl_backward()
        # so we don't require task_params vs shared_params split (avoids torchjd error).
        self.features = None
        # Add recursive_kld_loss (same KL fn, separate weight)
        self.objectives["recursive_kld_loss"] = self.objectives["kld_loss"]
        if isinstance(lambda_weights, list) and len(lambda_weights) >= 3:
            self.lambda_weights["recursive_kld_loss"] = lambda_weights[2]
        else:
            self.lambda_weights["recursive_kld_loss"] = 1.0

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
        mu, log_var = args["mu"], args["log_var"]
        mu_hat, log_var_hat = args["mu_hat"], args["log_var_hat"]

        recon_loss = self.objectives["reconstruction_loss"](inputs, recons)
        standard_kld = self.objectives["recursive_kld_loss"](mu, log_var)  # same KL fn
        recursive_kld = self.objectives["recursive_kld_loss"](mu_hat, log_var_hat)

        # Reconstruction loss = recon + lambda_kld * standard KL (not a separate loss)
        lambda_recon = self.lambda_weights["reconstruction_loss"]
        lambda_kld = self.lambda_weights["kld_loss"]  # weight for standard KL q(z|x)
        lambda_recursive_kld = self.lambda_weights["recursive_kld_loss"]  # weight for recursive KL q(z|hat_x)

        weighted_kld_loss = lambda_kld * standard_kld
        weighted_recon_loss = lambda_recon * recon_loss + weighted_kld_loss
        weighted_recursive_kld = lambda_recursive_kld * recursive_kld
        total_loss = weighted_recon_loss + weighted_recursive_kld

        return {
            "reconstruction_loss": weighted_recon_loss,
            # "kld_loss": weighted_kld_loss,
            "recursive_kld_loss": weighted_recursive_kld,
            "total_loss": total_loss,
        }
