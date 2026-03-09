"""
PixelCNN and PixelSNAIL Priors for VQ-VAE / VQ-VAE2
Learns the distribution of discrete latent codes for high-quality generation.
PixelSNAIL adds causal self-attention to PixelCNN for better long-range dependencies.
"""

from functools import lru_cache
from typing import Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@lru_cache(maxsize=32)
def _causal_attention_mask(size: int) -> Tensor:
    """Causal mask for raster-scan order: position i can attend to positions 0..i (include self).
    Returns [1, L, L] with 1 = attend, 0 = mask out. Allowing self-attention (diagonal) avoids
    NaN for the first position which would otherwise have no valid context."""
    mask = torch.tril(torch.ones(size, size))  # 1 where j <= i (lower triangle including diagonal)
    return mask.unsqueeze(0)  # [1, L, L]


class MaskedConv2d(nn.Conv2d):
    """
    Masked convolution for autoregressive modeling.
    Ensures that each pixel only depends on previously generated pixels.
    """
    def __init__(self, mask_type: str, *args, **kwargs):
        """
        Args:
            mask_type: 'A' or 'B'. 
                'A': First layer, excludes current pixel
                'B': Subsequent layers, includes current pixel
        """
        super().__init__(*args, **kwargs)
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"
        
        self.register_buffer('mask', torch.zeros_like(self.weight.data))
        
        _, _, kH, kW = self.weight.shape
        # Allow all rows before center
        self.mask[:, :, :kH // 2, :] = 1.0
        # Allow all columns before center in center row
        self.mask[:, :, kH // 2, :kW // 2] = 1.0
        
        # Type B includes center pixel
        if mask_type == 'B':
            self.mask[:, :, kH // 2, kW // 2] = 1.0
            
    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class GatedResBlock(nn.Module):
    """
    Gated Residual Block with masked convolutions.
    Uses gated activation: tanh(W_f * x) ⊙ sigmoid(W_g * x)
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        # Projection to smaller dimension
        self.conv1 = nn.Conv2d(channels, channels // 2, 1)
        
        # Masked convolution
        self.conv2 = MaskedConv2d('B', channels // 2, channels // 2, 
                                   kernel_size, padding=padding)
        
        # Gated activation (split into two paths)
        self.conv_gate = nn.Conv2d(channels // 2, channels, 1)
        self.conv_feature = nn.Conv2d(channels // 2, channels, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        
        # Gated activation
        gate = torch.sigmoid(self.conv_gate(out))
        feature = torch.tanh(self.conv_feature(out))
        out = gate * feature
        
        return residual + out


class CausalAttention2d(nn.Module):
    """
    Causal multi-head self-attention for 2D latent codes (raster-scan order).
    Position (i,j) can only attend to positions before it in row-major order.
    """
    def __init__(self, channels: int, num_heads: int = 8, head_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (channels // num_heads)
        assert channels % num_heads == 0 or head_dim is not None, "channels must be divisible by num_heads"
        self.proj_dim = self.head_dim * num_heads
        
        self.q_proj = nn.Conv2d(channels, self.proj_dim, 1)
        self.k_proj = nn.Conv2d(channels, self.proj_dim, 1)
        self.v_proj = nn.Conv2d(channels, self.proj_dim, 1)
        self.out_proj = nn.Conv2d(self.proj_dim, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        L = H * W
        # Flatten spatial: [B, C, H, W] -> [B, L, C]
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, L, C]
        
        q = self.q_proj(x).view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # [B, heads, L, head_dim]
        k = self.k_proj(x).view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # [B, heads, L, head_dim]
        v = self.v_proj(x).view(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # [B, heads, L, head_dim]
        
        # Attention: [B, heads, L, head_dim] @ [B, heads, head_dim, L] -> [B, heads, L, L]
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # Causal mask: mask out positions j >= i
        mask = _causal_attention_mask(L).to(dtype=attn.dtype, device=attn.device)
        attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, heads, L, head_dim]
        out = out.permute(0, 2, 3, 1).reshape(B, L, self.proj_dim).permute(0, 2, 1)
        out = out.view(B, self.proj_dim, H, W)
        return self.out_proj(out)


class PixelSNAILBlock(nn.Module):
    """
    PixelSNAIL block: residual blocks + causal self-attention.
    Interleaves causal convolutions with attention for unbounded receptive field.
    """
    def __init__(self, channels: int, num_res_blocks: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            GatedResBlock(channels) for _ in range(num_res_blocks)
        ])
        self.attention = CausalAttention2d(channels, num_heads=num_heads, dropout=dropout)
        self.out_conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.res_blocks:
            x = block(x)
        attn_out = self.attention(x)
        return self.out_conv(torch.cat([x, attn_out], dim=1)) + x


class PixelSNAIL(nn.Module):
    """
    PixelSNAIL: PixelCNN with causal self-attention for modeling discrete latent codes.
    Better captures long-range dependencies than PixelCNN alone.
    Drop-in replacement for PixelCNN with same interface.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int = 64,
                 hidden_channels: int = 128,
                 num_blocks: int = 8,
                 num_res_blocks_per_layer: int = 2,
                 num_heads: int = 8,
                 kernel_size: int = 7,
                 conditional_channels: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        input_channels = embedding_dim + conditional_channels + 2  # +2 for positional encoding
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.conv_in = MaskedConv2d('A', input_channels, hidden_channels,
                                    kernel_size, padding=kernel_size // 2)
        
        self.blocks = nn.ModuleList([
            PixelSNAILBlock(hidden_channels, num_res_blocks=num_res_blocks_per_layer,
                           num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        self.conv_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, num_embeddings, 1)
        )

    def _get_pos_encoding(self, height: int, width: int, device: torch.device) -> Tensor:
        """Row and col coordinates normalized to [-0.5, 0.5]. Returns [1, 2, H, W]."""
        coord_h = (torch.arange(height, device=device, dtype=torch.float32) - height / 2) / max(height, 1)
        coord_w = (torch.arange(width, device=device, dtype=torch.float32) - width / 2) / max(width, 1)
        pos_h = coord_h.view(1, 1, height, 1).expand(1, 1, height, width)
        pos_w = coord_w.view(1, 1, 1, width).expand(1, 1, height, width)
        return torch.cat([pos_h, pos_w], dim=1)

    def forward(self, x: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        B, H, W = x.shape
        x = self.embedding(x)  # [B, H, W, embedding_dim]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, embedding_dim, H, W]
        
        pos = self._get_pos_encoding(H, W, x.device).expand(B, -1, -1, -1)
        x = torch.cat([x, pos], dim=1)
        
        if condition is not None:
            x = torch.cat([x, condition], dim=1)
        
        x = self.conv_in(x)
        for block in self.blocks:
            x = x + block(x)
        logits = self.conv_out(x)
        return logits

    def sample(self,
               batch_size: int,
               height: int,
               width: int,
               device: torch.device,
               condition: Optional[Tensor] = None,
               temperature: float = 1.0) -> Tensor:
        self.eval()
        samples = torch.zeros(batch_size, height, width, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    logits = self(samples, condition)
                    logits = logits[:, :, i, j] / temperature
                    probs = F.softmax(logits, dim=1)
                    samples[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)
        return samples


class PixelCNN(nn.Module):
    """
    PixelCNN for modeling discrete latent code distributions.
    Can be used as a prior for VQ-VAE/VQ-VAE2.
    """
    def __init__(self, 
                 num_embeddings: int,
                 embedding_dim: int = 64,
                 hidden_channels: int = 128,
                 num_layers: int = 15,
                 kernel_size: int = 7,
                 conditional_channels: int = 0):
        """
        Args:
            num_embeddings: Size of the codebook (vocabulary size)
            embedding_dim: Dimension to embed discrete codes
            hidden_channels: Number of channels in hidden layers
            num_layers: Number of gated residual blocks
            kernel_size: Kernel size for first masked convolution
            conditional_channels: Number of channels for conditional input (for hierarchical modeling)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Embedding layer to convert discrete codes to continuous
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Input channels: embedding_dim + conditional_channels
        input_channels = embedding_dim + conditional_channels
        
        # First layer uses mask type 'A'
        self.conv_in = MaskedConv2d('A', input_channels, hidden_channels,
                                     kernel_size, padding=kernel_size // 2)
        
        # Stack of gated residual blocks
        self.res_blocks = nn.ModuleList([
            GatedResBlock(hidden_channels) for _ in range(num_layers)
        ])
        
        # Output layers
        self.conv_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, num_embeddings, 1)
        )
        
    def forward(self, x: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, H, W] - discrete code indices
            condition: [B, C, H, W] - optional conditional input
        Returns:
            logits: [B, num_embeddings, H, W] - probability distribution over codes
        """
        # Embed discrete codes
        x = self.embedding(x)  # [B, H, W, embedding_dim]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, embedding_dim, H, W]
        
        # Concatenate with condition if provided
        if condition is not None:
            x = torch.cat([x, condition], dim=1)
        
        # Forward through network
        x = self.conv_in(x)
        
        for block in self.res_blocks:
            x = block(x)
            
        logits = self.conv_out(x)
        
        return logits
    
    def sample(self, 
               batch_size: int,
               height: int,
               width: int,
               device: torch.device,
               condition: Optional[Tensor] = None,
               temperature: float = 1.0) -> Tensor:
        """
        Autoregressive sampling in raster scan order.
        
        Args:
            batch_size: Number of samples to generate
            height: Height of latent code spatial dimension
            width: Width of latent code spatial dimension
            device: Device to generate samples on
            condition: Optional conditioning input [B, C, H, W]
            temperature: Sampling temperature (higher = more diverse, lower = more conservative)
        
        Returns:
            samples: [B, H, W] - discrete code indices
        """
        self.eval()
        samples = torch.zeros(batch_size, height, width, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    # Get logits for current position
                    logits = self(samples, condition)  # [B, num_embeddings, H, W]
                    logits = logits[:, :, i, j] / temperature  # [B, num_embeddings]
                    
                    # Sample from categorical distribution
                    probs = F.softmax(logits, dim=1)
                    samples[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)
                    
        return samples


class HierarchicalPixelCNN(nn.Module):
    """
    Two-level PixelCNN for VQ-VAE2 hierarchical latent codes.
    Models P(z_top) and P(z_bottom | z_top).
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int = 64,
                 hidden_channels: int = 128,
                 num_layers: int = 15):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Prior for top-level codes: P(z_top)
        self.prior_top = PixelCNN(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )
        
        # Prior for bottom-level codes conditioned on top: P(z_bottom | z_top)
        # First, we need to process top codes to condition bottom
        self.embedding_top = nn.Embedding(num_embeddings, embedding_dim)
        self.upsample_top = nn.ConvTranspose2d(
            embedding_dim, embedding_dim,
            kernel_size=4, stride=2, padding=1
        )
        
        self.prior_bottom = PixelCNN(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            conditional_channels=embedding_dim  # Conditioned on upsampled top
        )

    def forward_top(self, z_top: Tensor) -> Tensor:
        return self.prior_top(z_top)

    def forward_bottom(self, z_bottom: Tensor, z_top: Tensor) -> Tensor:
        z_top_emb = self.embedding_top(z_top).permute(0, 3, 1, 2).contiguous()
        z_top_up = self.upsample_top(z_top_emb)
        return self.prior_bottom(z_bottom, condition=z_top_up)

    def forward(self, z_top: Tensor, z_bottom: Tensor) -> Dict[str, Tensor]:
        return {'logits_top': self.forward_top(z_top), 'logits_bottom': self.forward_bottom(z_bottom, z_top)}

    def loss_function(self, z_top: Tensor, z_bottom: Tensor) -> Dict[str, Tensor]:
        outputs = self.forward(z_top, z_bottom)
        loss_top = F.cross_entropy(
            outputs['logits_top'].permute(0, 2, 3, 1).reshape(-1, self.num_embeddings), z_top.reshape(-1))
        loss_bottom = F.cross_entropy(
            outputs['logits_bottom'].permute(0, 2, 3, 1).reshape(-1, self.num_embeddings), z_bottom.reshape(-1))
        return {'loss_top': loss_top, 'loss_bottom': loss_bottom, 'total_loss': loss_top + loss_bottom}

    def sample(self, batch_size: int, top_shape: tuple, bottom_shape: tuple, device: torch.device, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        self.eval()
        z_top = self.prior_top.sample(batch_size, top_shape[0], top_shape[1], device, temperature=temperature)
        z_top_emb = self.embedding_top(z_top).permute(0, 3, 1, 2).contiguous()
        z_top_up = self.upsample_top(z_top_emb)
        z_bottom = self.prior_bottom.sample(batch_size, bottom_shape[0], bottom_shape[1], device, condition=z_top_up, temperature=temperature)
        return z_top, z_bottom

    def sample_with_vqvae2(self, vqvae2_model, batch_size: int, device: torch.device, temperature: float = 1.0) -> Tensor:
        z_top, z_bottom = self.sample(batch_size,
            (vqvae2_model.latent_spatial_dim_top, vqvae2_model.latent_spatial_dim_top),
            (vqvae2_model.latent_spatial_dim_bottom, vqvae2_model.latent_spatial_dim_bottom),
            device, temperature)
        with torch.no_grad():
            quant_t = vqvae2_model.vq_top.embedding(z_top.view(batch_size, -1))
            quant_t = quant_t.view(batch_size, vqvae2_model.latent_spatial_dim_top, vqvae2_model.latent_spatial_dim_top, vqvae2_model.embedding_dim).permute(0, 3, 1, 2).contiguous()
            quant_b = vqvae2_model.vq_bottom.embedding(z_bottom.view(batch_size, -1))
            quant_b = quant_b.view(batch_size, vqvae2_model.latent_spatial_dim_bottom, vqvae2_model.latent_spatial_dim_bottom, vqvae2_model.embedding_dim).permute(0, 3, 1, 2).contiguous()
            return vqvae2_model.decode(quant_t, quant_b)

    def total_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HierarchicalPixelSNAIL(nn.Module):
    """
    Two-level prior for VQ-VAE2: PixelSNAIL (with attention) for top-level codes,
    PixelCNN for bottom-level (conditioned on top). Per VQ-VAE-2 paper, attention
    is used for global structure (top); bottom uses conditioning stacks for memory efficiency.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int = 64,
                 hidden_channels: int = 128,
                 num_blocks_top: int = 8,
                 num_res_blocks_per_layer: int = 2,
                 num_heads: int = 8,
                 num_layers_bottom: int = 15,
                 dropout: float = 0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.prior_top = PixelSNAIL(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks_top,
            num_res_blocks_per_layer=num_res_blocks_per_layer,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.embedding_top = nn.Embedding(num_embeddings, embedding_dim)
        self.upsample_top = nn.ConvTranspose2d(
            embedding_dim, embedding_dim,
            kernel_size=4, stride=2, padding=1
        )
        
        self.prior_bottom = PixelCNN(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers_bottom,
            conditional_channels=embedding_dim,
        )
        
    def forward_top(self, z_top: Tensor) -> Tensor:
        return self.prior_top(z_top)
        
    def forward_bottom(self, z_bottom: Tensor, z_top: Tensor) -> Tensor:
        z_top_emb = self.embedding_top(z_top).permute(0, 3, 1, 2).contiguous()
        z_top_up = self.upsample_top(z_top_emb)
        return self.prior_bottom(z_bottom, condition=z_top_up)
    
    def forward(self, z_top: Tensor, z_bottom: Tensor) -> Dict[str, Tensor]:
        return {
            'logits_top': self.forward_top(z_top),
            'logits_bottom': self.forward_bottom(z_bottom, z_top),
        }
    
    def loss_function(self, z_top: Tensor, z_bottom: Tensor) -> Dict[str, Tensor]:
        outputs = self.forward(z_top, z_bottom)
        loss_top = F.cross_entropy(
            outputs['logits_top'].permute(0, 2, 3, 1).reshape(-1, self.num_embeddings),
            z_top.reshape(-1),
        )
        loss_bottom = F.cross_entropy(
            outputs['logits_bottom'].permute(0, 2, 3, 1).reshape(-1, self.num_embeddings),
            z_bottom.reshape(-1),
        )
        return {
            'loss_top': loss_top,
            'loss_bottom': loss_bottom,
            'total_loss': loss_top + loss_bottom,
        }
    
    def sample(self,
               batch_size: int,
               top_shape: tuple,
               bottom_shape: tuple,
               device: torch.device,
               temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        z_top = self.prior_top.sample(
            batch_size, top_shape[0], top_shape[1], device, temperature=temperature
        )
        z_top_emb = self.embedding_top(z_top).permute(0, 3, 1, 2).contiguous()
        z_top_up = self.upsample_top(z_top_emb)
        z_bottom = self.prior_bottom.sample(
            batch_size, bottom_shape[0], bottom_shape[1], device,
            condition=z_top_up, temperature=temperature
        )
        return z_top, z_bottom
    
    def sample_with_vqvae2(self,
                           vqvae2_model,
                           batch_size: int,
                           device: torch.device,
                           temperature: float = 1.0) -> Tensor:
        z_top, z_bottom = self.sample(
            batch_size,
            (vqvae2_model.latent_spatial_dim_top, vqvae2_model.latent_spatial_dim_top),
            (vqvae2_model.latent_spatial_dim_bottom, vqvae2_model.latent_spatial_dim_bottom),
            device,
            temperature,
        )
        with torch.no_grad():
            quant_t = vqvae2_model.vq_top.embedding(z_top.view(batch_size, -1))
            quant_t = quant_t.view(
                batch_size,
                vqvae2_model.latent_spatial_dim_top,
                vqvae2_model.latent_spatial_dim_top,
                vqvae2_model.embedding_dim
            ).permute(0, 3, 1, 2).contiguous()
            quant_b = vqvae2_model.vq_bottom.embedding(z_bottom.view(batch_size, -1))
            quant_b = quant_b.view(
                batch_size,
                vqvae2_model.latent_spatial_dim_bottom,
                vqvae2_model.latent_spatial_dim_bottom,
                vqvae2_model.embedding_dim
            ).permute(0, 3, 1, 2).contiguous()
            return vqvae2_model.decode(quant_t, quant_b)
    
    def total_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

