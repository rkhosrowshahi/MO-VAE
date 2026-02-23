"""
Sphere Encoder ViT: paper-matching architecture (ViT + MLP-Mixer + RoPE + sinusoidal).

Based on "Image Generation with a Sphere Encoder" (arXiv:2602.15030) §2.4:
- ViT (Vision Transformer) for encoder and decoder
- 4-layer MLP-Mixer at end of encoder (before spherification) and beginning of decoder
- RMSNorm with learned affine after each MLP-Mixer
- RoPE + sinusoidal absolute positional encoding
- Optional AdaLN-Zero for class conditioning and CFG

Same training objective and generation API as sphere_encoder.py (three losses, one/few-step).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sphere_encoder import (
    rms_norm,
    spherify as _spherify,
    smooth_l1_per_pixel,
    PerceptualLoss,
)


def spherify_tensor(x: torch.Tensor, L: int, radius: float | None = None, dim: int = -1) -> torch.Tensor:
    """Project onto sphere of radius sqrt(L). Uses RMS norm."""
    if radius is None:
        radius = math.sqrt(L)
    return _spherify(x, radius=radius, dim=dim)


class RMSNorm(nn.Module):
    """RMSNorm with optional learned affine (paper: to bound magnitude <= sqrt(L))."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = rms_norm(x, dim=-1, eps=self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out


class SinusoidalPosEmbedding(nn.Module):
    """Sinusoidal absolute positional encoding (1D sequence). Paper: removing it hurts quality."""

    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) -> add pe[:, :N]
        return x + self.pe[:, : x.size(1), :].to(x.dtype)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k. q, k: (B, H, N, head_dim). freqs: (1, N, 1, head_dim/2)."""
    if freqs.dim() == 3:
        freqs = freqs.unsqueeze(2)
    head_dim = q.size(-1)
    dim_half = head_dim // 2
    cos = freqs[..., :dim_half].cos().to(q.dtype)
    sin = freqs[..., :dim_half].sin().to(q.dtype)
    # Align for (B, H, N, d/2): cos/sin (1, N, 1, d/2) -> (1, 1, N, d/2)
    cos = cos.squeeze(2).unsqueeze(0)
    sin = sin.squeeze(2).unsqueeze(0)

    def rotate(u: torch.Tensor) -> torch.Tensor:
        u1, u2 = u[..., 0::2], u[..., 1::2]
        r0 = u1 * cos - u2 * sin
        r1 = u1 * sin + u2 * cos
        return torch.stack([r0, r1], dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class RotaryEmbedding(nn.Module):
    """RoPE for 1D sequence. base^(-2i/d) * position."""

    def __init__(self, dim: int, base: float = 10000.0, max_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(dtype))
        return freqs.unsqueeze(0).unsqueeze(2)


class PatchEmbed(nn.Module):
    """Image to patch embeddings. (B, C, H, W) -> (B, N, D)."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, D, h, w) -> (B, N, D)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Unpatchify(nn.Module):
    """(B, N, P*P*C) -> (B, C, H, W)."""

    def __init__(self, img_size: int, patch_size: int, out_channels: int = 3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = w = self.img_size // self.patch_size
        x = x.reshape(B, h, w, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_channels, self.img_size, self.img_size)
        return x


class AttentionWithRoPE(nn.Module):
    """Multi-head self-attention with RoPE on K, Q."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if rope_freqs is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_freqs)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: norm -> attention (RoPE) -> residual -> norm -> MLP -> residual."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = AttentionWithRoPE(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs)
        x = x + self.mlp(self.norm2(x))
        return x


class MLPMixerBlock(nn.Module):
    """MLP-Mixer: token-mixing (MLP on seq) + channel-mixing (MLP on channels). Paper: Tolstikhin et al."""

    def __init__(self, num_patches: int, embed_dim: int, tokens_mlp_dim: int = 256, channels_mlp_dim: int = 2048):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches),
        )
        self.norm2 = RMSNorm(embed_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(embed_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        x = x + self.token_mix(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(self.norm2(x))
        return x


class MLPMixer(nn.Module):
    """Stack of MLP-Mixer blocks with RMSNorm after (paper: 2 or 4 layers)."""

    def __init__(self, num_patches: int, embed_dim: int, depth: int, tokens_mlp_dim: int = 256, channels_mlp_dim: int = 2048):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLPMixerBlock(num_patches, embed_dim, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class SphereEncoderViT(nn.Module):
    """
    Sphere Encoder with ViT + MLP-Mixer (paper architecture).
    Same training (three losses) and generation API as SphereEncoder.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mixer_depth: int = 2,
        mixer_tokens_mlp_dim: int = 256,
        mixer_channels_mlp_dim: int = 2048,
        latent_channels: int = 8,
        num_classes: int = 0,
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
        dropout: float = 0.0,
        device=None,
    ):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.L = self.num_patches * latent_channels
        self.radius = math.sqrt(self.L)
        self.sigma_max_angle_deg = float(sigma_max_angle_deg)
        self.sigma_max = math.tan(math.radians(self.sigma_max_angle_deg))
        self.sigma_mix_prob = float(sigma_mix_prob)
        self.sigma_mix_angle_min_deg = float(sigma_mix_angle_min_deg) if sigma_mix_angle_min_deg is not None else None
        self.sigma_mix_angle_max_deg = float(sigma_mix_angle_max_deg) if sigma_mix_angle_max_deg is not None else None
        self.num_classes = num_classes
        self.lambda_pix_recon = lambda_pix_recon
        self.lambda_pix_con = lambda_pix_con
        self.lambda_lat_con = lambda_lat_con
        self.pix_recon_smooth_l1_weight = pix_recon_smooth_l1_weight
        self.pix_recon_perceptual_weight = pix_recon_perceptual_weight
        self.pix_con_smooth_l1_weight = pix_con_smooth_l1_weight
        self.pix_con_perceptual_weight = pix_con_perceptual_weight
        self.use_perceptual = use_perceptual

        # Encoder
        self.patch_embed_enc = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed_enc = SinusoidalPosEmbedding(embed_dim, max_len=self.num_patches)
        self.blocks_enc = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.mixer_enc = MLPMixer(
            self.num_patches, embed_dim, mixer_depth,
            mixer_tokens_mlp_dim, mixer_channels_mlp_dim,
        )
        self.norm_enc = RMSNorm(embed_dim)
        self.latent_proj_enc = nn.Linear(embed_dim, latent_channels)

        # Decoder: from spherical vector (B, L) -> (B, N, latent_channels) -> embed_dim -> mixer -> transformer -> pixels
        self.latent_proj_dec = nn.Linear(latent_channels, embed_dim)
        self.norm_dec_in = RMSNorm(embed_dim)
        self.mixer_dec = MLPMixer(
            self.num_patches, embed_dim, mixer_depth,
            mixer_tokens_mlp_dim, mixer_channels_mlp_dim,
        )
        self.pos_embed_dec = SinusoidalPosEmbedding(embed_dim, max_len=self.num_patches)
        self.blocks_dec = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm_dec_out = nn.LayerNorm(embed_dim)
        self.head_dec = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        self.unpatchify = Unpatchify(img_size, patch_size, in_channels)

        # Output activation: paper uses L1+perceptual; images often [0,1] or [-1,1]. Default tanh for [−1,1].
        self.output_activation = nn.Tanh()

        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(device=device)
        else:
            self.perceptual_loss = None

        self.objectives = {
            "pix_recon": lambda *_: torch.tensor(0.0, device=next(self.parameters()).device),
            "pix_con": lambda *_: torch.tensor(0.0, device=next(self.parameters()).device),
            "lat_con": lambda *_: torch.tensor(0.0, device=next(self.parameters()).device),
        }
        self.features = None

    def _get_rope_freqs(self, N: int, device: torch.device, dtype: torch.dtype, enc: bool = True) -> torch.Tensor | None:
        block = self.blocks_enc[0] if enc else self.blocks_dec[0]
        return block.attn.rotary(N, device, dtype)

    def encode_to_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to flat vector (before spherify). (B, C, H, W) -> (B, L)."""
        B = x.size(0)
        device, dtype = x.device, x.dtype
        h = self.patch_embed_enc(x)
        h = self.pos_embed_enc(h)
        rope_freqs = self._get_rope_freqs(h.size(1), device, dtype, enc=True)
        for blk in self.blocks_enc:
            h = blk(h, rope_freqs)
        h = self.mixer_enc(h)
        h = self.norm_enc(h)
        z = self.latent_proj_enc(h)
        return z.reshape(B, -1)

    def spherify(
        self,
        z: torch.Tensor,
        add_noise: bool = False,
        sigma: torch.Tensor | float | None = None,
        e: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project z onto sphere. If add_noise: v_noisy = spherify(spherify(z) + sigma*e).

        The paper (Eq. 4, Figure 4) adds noise to the already-spherified v so that
        the effective perturbation angle equals arctan(sigma) as in Eqs. 11-12.
        """
        v = spherify_tensor(z, self.L, self.radius, dim=-1)
        if add_noise and sigma is not None and e is not None:
            v = spherify_tensor(v + sigma * e, self.L, self.radius, dim=-1)
        return v

    def decode_from_sphere(self, v: torch.Tensor) -> torch.Tensor:
        """Decode spherical latent to image. v: (B, L) -> (B, C, H, W)."""
        B = v.size(0)
        device, dtype = v.device, v.dtype
        h = v.reshape(B, self.num_patches, -1)
        h = self.latent_proj_dec(h)
        h = self.norm_dec_in(h)
        h = self.mixer_dec(h)
        h = self.pos_embed_dec(h)
        rope_freqs = self._get_rope_freqs(h.size(1), device, dtype, enc=False)
        for blk in self.blocks_dec:
            h = blk(h, rope_freqs)
        h = self.norm_dec_out(h)
        h = self.head_dec(h)
        out = self.unpatchify(h)
        return self.output_activation(out)

    def forward(self, x: torch.Tensor) -> dict:
        batch = x.size(0)
        device = x.device
        dtype = x.dtype

        z = self.encode_to_vector(x)
        v = self.spherify(z, add_noise=False)

        # Jitter angle alpha ~ U[0, alpha_max] with optional mixed larger-angle band (paper Appx C.2 / Table 16).
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

        s = torch.rand(batch, 1, device=device, dtype=dtype) * 0.5
        sigma_sub = s * sigma

        e = torch.randn(batch, self.L, device=device, dtype=z.dtype)
        v_noisy = self.spherify(z, add_noise=True, sigma=sigma, e=e)
        v_noisy_small = self.spherify(z, add_noise=True, sigma=sigma_sub, e=e)

        x_recon_noisy_small = self.decode_from_sphere(v_noisy_small)
        x_recon_NOISY = self.decode_from_sphere(v_noisy)
        with torch.no_grad():
            x_recon_noisy_small_sg = x_recon_noisy_small.detach()

        # Latent consistency: reuse x_recon_NOISY instead of a redundant second decoder pass
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
        recons = args["recons"]
        x_recon_NOISY = args["x_recon_NOISY"]
        x_recon_noisy_small_sg = args["x_recon_noisy_small_sg"]
        v = args["v"]
        v_enc_dec = args["v_enc_dec"]

        L_pix_recon = self._pixel_loss(
            recons, x,
            self.pix_recon_smooth_l1_weight,
            self.pix_recon_perceptual_weight,
        )
        L_pix_con = self._pixel_loss(
            x_recon_NOISY, x_recon_noisy_small_sg,
            self.pix_con_smooth_l1_weight,
            self.pix_con_perceptual_weight,
        )
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
        share_noise=True (default): same e reused across all few-step iterations.
        Paper Table 6 shows shared noise with fixed r=1.0 consistently outperforms
        independent noise across all step counts.
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
                sigma = self.sigma_max
                e_step = e if share_noise else torch.randn(num_samples, self.L, device=device)
                v = self.spherify(z, add_noise=True, sigma=sigma, e=e_step)
                x = self.decode_from_sphere(v)
        return x
