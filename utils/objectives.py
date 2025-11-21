import torch
from torch.nn import functional as F


# Mean squared error loss with sum reduction
# Standard VAE Loss (Sum over features, Mean over batch)
def mse_per_image_sum(inputs, recons):
    loss = F.mse_loss(recons, inputs, reduction='sum') / inputs.size(0)
    return loss

# Scaled Total Loss (Sum over features, Sum over batch, * 255)
def mse_total_batch_sum_scaled(inputs, recons):
    loss = F.mse_loss(recons * 255.0, inputs * 255.0, reduction='sum') / 255.0
    return loss

# Mean squared error loss with mean reduction
# Per-Pixel Loss (Mean over features, Mean over batch)
def mse_per_pixel_mean(inputs, recons):
    loss = F.mse_loss(recons, inputs, reduction='mean')
    return loss

# Binary cross entropy loss
# NOTE: The VAE decoder uses a sigmoid output activation when a BCE objective is selected,
# so we use `binary_cross_entropy` (which expects probabilities) rather than
# `binary_cross_entropy_with_logits` (which expects raw logits).
def bce_per_image_sum(inputs, recons):
    loss = F.binary_cross_entropy(recons, inputs, reduction='sum') / inputs.size(0)
    return loss

# Binary cross entropy loss with mean reduction
def bce_per_pixel_mean(inputs, recons):
    loss = F.binary_cross_entropy(recons, inputs, reduction='mean')
    return loss

def bce_with_logits_per_image_sum(inputs, recons):
    loss = F.binary_cross_entropy_with_logits(recons, inputs, reduction='sum') / inputs.size(0)
    return loss

# Binary cross entropy loss with mean reduction
def bce_with_logits_per_pixel_mean(inputs, recons):
    loss = F.binary_cross_entropy_with_logits(recons, inputs, reduction='mean')
    return loss

# Laplacian loss (L1 loss) with sum reduction
# For Laplacian distribution: p(x|z) = Laplacian(recons, scale)
# Negative log-likelihood is proportional to L1 loss
def laplacian_per_image_sum(inputs, recons):
    loss = F.l1_loss(recons, inputs, reduction='sum') / inputs.size(0)
    return loss

# Laplacian loss (L1 loss) with mean reduction
def laplacian_per_pixel_mean(inputs, recons):
    loss = F.l1_loss(recons, inputs, reduction='mean')
    return loss

# KL divergence loss (mean over batch)
# Formula: D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
# where q(z|x) ~ N(mu, exp(log_var)) and p(z) ~ N(0, I)
def kl_divergence(mu, log_var):
    # Sum over latent dimensions, then average over batch
    kl_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return kl_per_sample.mean()