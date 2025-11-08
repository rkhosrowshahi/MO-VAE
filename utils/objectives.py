import torch
from torch.nn import functional as F


# Mean squared error loss with sum reduction
def mse_recon_sum(inputs, recons):
    loss = F.mse_loss(recons, inputs, reduction='sum') / inputs.size(0)
    return loss

# Mean squared error loss with mean reduction
def mse_recon_mean(inputs, recons):
    loss = F.mse_loss(recons, inputs, reduction='mean')
    return loss

# Binary cross entropy loss
def bce_recon_sum(inputs, recons):
    loss = F.binary_cross_entropy_with_logits(recons, inputs, reduction='sum') / inputs.size(0)
    return loss

# Binary cross entropy loss with mean reduction
def bce_recon_mean(inputs, recons):
    loss = F.binary_cross_entropy_with_logits(recons, inputs, reduction='mean')
    return loss

# KL divergence loss (mean over batch)
# Formula: D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
# where q(z|x) ~ N(mu, exp(log_var)) and p(z) ~ N(0, I)
def kl_divergence_mean(mu, log_var):
    # Sum over latent dimensions, then average over batch
    kl_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return kl_per_sample.mean()

# KL divergence loss (sum over batch) - for consistency with sum reduction in reconstruction loss
def kl_divergence_sum(mu, log_var):
    # Sum over latent dimensions and batch
    kl_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return kl_per_sample.sum()