"""
Sphere Encoder: Image generation with spherical latent space.

Based on "Image Generation with a Sphere Encoder" (arXiv:2602.15030).
- Encoder maps images to a vector that is projected onto a sphere (RMS normalization).
- Decoder maps points on the sphere back to images.
- Trained with reconstruction + pixel consistency + latent consistency (no KLD).
- Generation: decode a random point on the sphere (one-step) or iterate encode/decode (few-step).

This implementation uses a Conv backbone (same as VAE) for compatibility with small images;
the paper uses ViT for 256x256. Spherify, noise schedule, and three losses follow the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt

from .vae import VAE


def rms_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Root mean square normalization: x / rms(x)."""
    rms = (x.pow(2).mean(dim=dim, keepdim=True) + eps).sqrt()
    return x / rms


def spherify(x: torch.Tensor, radius: float | None = None, dim: int = -1) -> torch.Tensor:
    """
    Project vector(s) onto sphere of radius sqrt(L) via RMS normalization.
    v = sqrt(L) * x / rms(x) so that ||v|| = sqrt(L).
    """
    L = x.shape[dim]
    if radius is None:
        radius = sqrt(L)
    v = rms_norm(x, dim=dim) * radius
    return v


def smooth_l1_per_pixel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 (Huber) loss, mean over pixels and batch."""
    return F.smooth_l1_loss(pred, target, reduction="mean")


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


class SphereEncoder(VAE):
    """
    Sphere Encoder: encoder maps images to sphere, decoder maps sphere to images.
    Trained with L_pix_recon + L_pix_con + L_lat_con (no KLD).
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        sigma_max_angle_deg: float = 80.0,
        sigma_mix_prob: float = 0.0,
        sigma_mix_angle_min_deg: float | None = None,
        sigma_mix_angle_max_deg: float | None = None,
        lambda_pix_recon: float = 1.0,
        lambda_pix_con: float = 0.5,
        lambda_lat_con: float = 0.1,
        pix_recon_smooth_l1_weight: float = 1.0,
        pix_recon_perceptual_weight: float = 1.0,
        pix_con_smooth_l1_weight: float = 0.5,
        pix_con_perceptual_weight: float = 0.5,
        use_perceptual: bool = True,
        **kwargs,
    ):
        # Build base VAE encoder/decoder; we override mu/log_var and loss.
        # Base VAE expects latent_dim and creates mu, log_var; we use latent_dim as sphere dim L.
        super().__init__(latent_dim=latent_dim, **kwargs)

        # Remove VAE latent heads; we use a single linear to L and spherify
        encoder_output_size = self.hidden_dims[-1] * (self.input_size // (2 ** len(self.hidden_dims))) ** 2
        del self.mu
        del self.log_var
        self.encoder_proj = nn.Linear(encoder_output_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, encoder_output_size)

        self.L = latent_dim
        self.radius = sqrt(latent_dim)
        # Noise magnitude is controlled by an angle alpha on the sphere (paper Appx D/C.2).
        # We keep sigma_max = tan(alpha_max) for inference-time fixed-strength sampling.
        import math
        self.sigma_max_angle_deg = float(sigma_max_angle_deg)
        self.sigma_max = math.tan(math.radians(self.sigma_max_angle_deg))
        self.sigma_mix_prob = float(sigma_mix_prob)
        self.sigma_mix_angle_min_deg = float(sigma_mix_angle_min_deg) if sigma_mix_angle_min_deg is not None else None
        self.sigma_mix_angle_max_deg = float(sigma_mix_angle_max_deg) if sigma_mix_angle_max_deg is not None else None
        self.lambda_pix_recon = lambda_pix_recon
        self.lambda_pix_con = lambda_pix_con
        self.lambda_lat_con = lambda_lat_con
        self.pix_recon_smooth_l1_weight = pix_recon_smooth_l1_weight
        self.pix_recon_perceptual_weight = pix_recon_perceptual_weight
        self.pix_con_smooth_l1_weight = pix_con_smooth_l1_weight
        self.pix_con_perceptual_weight = pix_con_perceptual_weight
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(device=getattr(self, "device", None))
        else:
            self.perceptual_loss = None

        # Objectives for MTL / logging (no kld)
        self.objectives = {
            "pix_recon": lambda *_: torch.tensor(0.0, device=next(self.parameters()).device),
            "pix_con": lambda *_: torch.tensor(0.0, device=next(self.parameters()).device),
            "lat_con": lambda *_: torch.tensor(0.0, device=next(self.parameters()).device),
        }
        self.features = None  # no feature-based MTL

    def encode_to_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to flat vector (before spherify)."""
        h = self.encoder(x)
        z = self.encoder_proj(h)
        return z

    def spherify(
        self,
        z: torch.Tensor,
        add_noise: bool = False,
        sigma: torch.Tensor | float | None = None,
        e: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project z onto sphere. If add_noise: v_noisy = spherify(spherify(z) + sigma*e).

        The paper (Eq. 4, Figure 4) says noise is added to the already-spherified v,
        not to the raw encoder output z. We therefore spherify first, then perturb v,
        then re-spherify, so the effective angle is arctan(sigma) as per Eqs. 11-12.
        """
        v = spherify(z, radius=self.radius, dim=-1)
        if add_noise and sigma is not None and e is not None:
            v = spherify(v + sigma * e, radius=self.radius, dim=-1)
        return v

    def decode_from_sphere(self, v: torch.Tensor) -> torch.Tensor:
        """Decode from spherical latent v (vector on sphere)."""
        h = self.decoder_input(v)
        out = self.decoder(h)
        recons = self.final_layer(out)
        return recons

    def encode(self, x: torch.Tensor):
        """Return (v,) on sphere for compatibility; no mu/log_var."""
        z = self.encode_to_vector(x)
        v = self.spherify(z, add_noise=False)
        return (v,)

    def reparameterize(self, mu, log_var):
        """Not used; sphere encoder has no stochastic reparameterization."""
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent (can be spherical vector or unprojected)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        # If already on sphere (norm ~ radius), decode directly; else spherify first
        norm = z.norm(dim=-1, keepdim=True)
        if not torch.allclose(norm, torch.full_like(norm, self.radius), atol=1e-2):
            z = self.spherify(z, add_noise=False)
        return self.decode_from_sphere(z)

    def forward(self, x: torch.Tensor) -> dict:
        batch = x.size(0)
        device = x.device
        dtype = x.dtype

        # Encode and spherify (clean)
        z = self.encode_to_vector(x)
        v = self.spherify(z, add_noise=False)

        # Jitter angle alpha ~ U[0, alpha_max] (paper Table 16 / Appx C.2).
        # Optional "mix band" of larger angles with probability p (paper Appx C.2).
        angle_max = self.sigma_max_angle_deg
        angle_deg = torch.rand(batch, 1, device=device, dtype=dtype) * angle_max
        if self.sigma_mix_prob > 0 and self.sigma_mix_angle_min_deg is not None and self.sigma_mix_angle_max_deg is not None:
            mix_min = self.sigma_mix_angle_min_deg
            mix_max = self.sigma_mix_angle_max_deg
            if mix_max > mix_min:
                mix_mask = (torch.rand(batch, 1, device=device) < self.sigma_mix_prob)
                mix_angle = mix_min + torch.rand(batch, 1, device=device, dtype=dtype) * (mix_max - mix_min)
                angle_deg = torch.where(mix_mask, mix_angle, angle_deg)
        angle_rad = angle_deg * (math.pi / 180.0)
        sigma = torch.tan(angle_rad)

        # sigma_sub = s * sigma, where s ~ U[0, 0.5], and shares the same noise direction e.
        s = torch.rand(batch, 1, device=device, dtype=dtype) * 0.5
        sigma_sub = s * sigma

        e = torch.randn(batch, self.L, device=device, dtype=z.dtype)
        v_noisy = self.spherify(z, add_noise=True, sigma=sigma, e=e)
        v_noisy_small = self.spherify(z, add_noise=True, sigma=sigma_sub, e=e)

        x_recon_noisy_small = self.decode_from_sphere(v_noisy_small)
        x_recon_NOISY = self.decode_from_sphere(v_noisy)
        with torch.no_grad():
            x_recon_noisy_small_sg = x_recon_noisy_small.detach()

        # Latent consistency: encode decoded noisy image (reuse x_recon_NOISY, no second decoder pass)
        z_enc_dec = self.encode_to_vector(x_recon_NOISY)
        v_enc_dec = self.spherify(z_enc_dec, add_noise=False)

        return {
            "recons": x_recon_noisy_small,
            "v": v,
            "v_noisy": v_noisy,
            "v_noisy_small": v_noisy_small,
            "x_recon_NOISY": x_recon_NOISY,
            "x_recon_noisy_small_sg": x_recon_noisy_small_sg,
            "v_enc_dec": v_enc_dec,
            "sigma": sigma,
            "sigma_sub": sigma_sub,
        }

    def _pixel_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth_l1_w: float, perceptual_w: float) -> torch.Tensor:
        loss = smooth_l1_w * smooth_l1_per_pixel(pred, target)
        if self.use_perceptual and self.perceptual_loss is not None and perceptual_w > 0:
            loss = loss + perceptual_w * self.perceptual_loss(pred, target)
        return loss

    def loss_function(self, inputs: torch.Tensor, args: dict) -> dict:
        x = inputs
        recons = args["recons"]           # D(v_noisy_small)
        x_recon_NOISY = args["x_recon_NOISY"]  # D(v_NOISY)
        x_recon_noisy_small_sg = args["x_recon_noisy_small_sg"]  # sg(D(v_noisy_small))
        v = args["v"]
        v_enc_dec = args["v_enc_dec"]

        # L_pix_recon: reconstruct x from v_noisy_small
        L_pix_recon = self._pixel_loss(
            recons, x,
            self.pix_recon_smooth_l1_weight,
            self.pix_recon_perceptual_weight,
        )
        # L_pix_con: D(v_NOISY) should match sg(D(v_noisy_small))
        L_pix_con = self._pixel_loss(
            x_recon_NOISY, x_recon_noisy_small_sg,
            self.pix_con_smooth_l1_weight,
            self.pix_con_perceptual_weight,
        )
        # L_lat_con: cosine similarity between v and E(D(v_NOISY))
        cos_sim = F.cosine_similarity(v, v_enc_dec, dim=-1)
        L_lat_con = (1 - cos_sim).mean()

        weighted_pix_recon = self.lambda_pix_recon * L_pix_recon
        weighted_pix_con = self.lambda_pix_con * L_pix_con
        weighted_lat_con = self.lambda_lat_con * L_lat_con
        total_loss = weighted_pix_recon + weighted_pix_con + weighted_lat_con

        return {
            "pix_recon": weighted_pix_recon,
            "pix_con": weighted_pix_con,
            "lat_con": weighted_lat_con,
            "total_loss": total_loss,
        }

    def sample(self, num_samples: int = 1, device=None, steps: int = 1, share_noise: bool = True) -> torch.Tensor:
        """
        Generate images: one-step (decode random point on sphere) or few-step (iterate encode-decode).
        steps=1: x = D(spherify(e)), e ~ N(0,I).
        steps>1: iterate encode-decode, adding noise before re-spherifying.

        share_noise=True (default): same e used across all steps (paper Table 6 shows this
        consistently outperforms independent noise with fixed r=1.0 schedule).
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            e = torch.randn(num_samples, self.L, device=device)
            v = self.spherify(e, add_noise=False)
            x = self.decode_from_sphere(v)
            for _ in range(steps - 1):
                z = self.encode_to_vector(x)
                # Paper: fixed r=1.0 for few-step at inference (§2.3, Algorithm 1)
                sigma = self.sigma_max
                e_step = e if share_noise else torch.randn(num_samples, self.L, device=device)
                v = self.spherify(z, add_noise=True, sigma=sigma, e=e_step)
                x = self.decode_from_sphere(v)
        return x
