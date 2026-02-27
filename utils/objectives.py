import torch
import torch.nn as nn
from torch.nn import functional as F


def get_recon_obj_and_activation(recons_objective, recons_activation="tanh", model=None, use_logits=False):
    """
    Build reconstruction objective and activation from recons_objective string.
    recons_objective: one of "mse", "bce", "l1", "smooth_l1", "perceptual"
    model: required when recons_objective="perceptual" to register PerceptualLoss submodule
    use_logits: if True and recons_objective="bce", use BCE-with-logits (for decoders without sigmoid)
    Returns: (recon_fn, recons_activation)
    """
    recons_objective = recons_objective.lower()
    valid = {"mse", "bce", "l1", "smooth_l1", "perceptual"}
    if recons_objective not in valid:
        raise ValueError(f"recons_objective must be one of {valid}, got {recons_objective}")

    if recons_objective == "mse":
        recon_obj = mse_per_pixel_mean
        recons_activation = recons_activation or "tanh"
    elif recons_objective == "bce":
        if use_logits:
            recon_obj = bce_with_logits_per_pixel_mean
            recons_activation = "none"  # decoder outputs logits
        else:
            recon_obj = bce_per_pixel_mean
            recons_activation = "sigmoid"
    elif recons_objective == "l1":
        recon_obj = laplacian_per_pixel_mean
        recons_activation = recons_activation or "tanh"
    elif recons_objective == "smooth_l1":
        recon_obj = smooth_l1_per_pixel_mean
        recons_activation = recons_activation or "tanh"
    elif recons_objective == "perceptual":
        if model is None:
            raise ValueError("model required for recons_objective='perceptual' to register PerceptualLoss")
        device = getattr(model, "device", None)
        pl = PerceptualLoss(device=device)
        model.perceptual_loss = pl  # register submodule so it moves with model
        recon_obj = make_perceptual_recon_fn(pl)
        recons_activation = recons_activation or "tanh"
    return recon_obj, recons_activation


def make_perceptual_recon_fn(perceptual_loss_module):
    """Return a (inputs, recons) -> loss function that uses the given PerceptualLoss module."""
    def fn(inputs, recons):
        return perceptual_loss_module(recons, inputs)
    return fn


class PerceptualLoss(nn.Module):
    """Differentiable perceptual loss using VGG16 features (L2 on features)."""

    def __init__(self, device=None):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])  # up to conv3_3
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.device = device

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() < 0:
            x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._norm_input(pred)
        target_n = self._norm_input(target)
        f_pred = self.features(pred_n)
        f_target = self.features(target_n)
        return F.mse_loss(f_pred, f_target)


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

# Smooth L1 (Huber) loss with mean reduction
def smooth_l1_per_pixel_mean(inputs, recons):
    loss = F.smooth_l1_loss(recons, inputs, reduction='mean')
    return loss

# KL divergence loss (mean over batch)
# Formula: D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
# where q(z|x) ~ N(mu, exp(log_var)) and p(z) ~ N(0, I)
def kl_divergence(mu, log_var):
    # Sum over latent dimensions, then average over batch
    kl_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return kl_per_sample.mean()