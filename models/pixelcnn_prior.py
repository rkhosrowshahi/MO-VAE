"""
PixelCNN Prior for VQ-VAE2
Learns the distribution of discrete latent codes for high-quality generation.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
        """
        Compute logits for top-level codes.
        
        Args:
            z_top: [B, H_t, W_t] - discrete code indices for top level
        Returns:
            logits: [B, num_embeddings, H_t, W_t]
        """
        return self.prior_top(z_top)
    
    def forward_bottom(self, z_bottom: Tensor, z_top: Tensor) -> Tensor:
        """
        Compute logits for bottom-level codes conditioned on top.
        
        Args:
            z_bottom: [B, H_b, W_b] - discrete code indices for bottom level
            z_top: [B, H_t, W_t] - discrete code indices for top level
        Returns:
            logits: [B, num_embeddings, H_b, W_b]
        """
        # Embed and upsample top codes to condition bottom
        z_top_emb = self.embedding_top(z_top)  # [B, H_t, W_t, D]
        z_top_emb = z_top_emb.permute(0, 3, 1, 2).contiguous()  # [B, D, H_t, W_t]
        z_top_up = self.upsample_top(z_top_emb)  # [B, D, H_b, W_b]
        
        # Forward through conditional PixelCNN
        return self.prior_bottom(z_bottom, condition=z_top_up)
    
    def forward(self, z_top: Tensor, z_bottom: Tensor) -> Dict[str, Tensor]:
        """
        Compute logits for both levels.
        
        Args:
            z_top: [B, H_t, W_t] - top-level code indices
            z_bottom: [B, H_b, W_b] - bottom-level code indices
        Returns:
            Dictionary with 'logits_top' and 'logits_bottom'
        """
        logits_top = self.forward_top(z_top)
        logits_bottom = self.forward_bottom(z_bottom, z_top)
        
        return {
            'logits_top': logits_top,
            'logits_bottom': logits_bottom
        }
    
    def loss_function(self, z_top: Tensor, z_bottom: Tensor) -> Dict[str, Tensor]:
        """
        Compute cross-entropy losses for both levels.
        
        Args:
            z_top: [B, H_t, W_t] - ground truth top codes
            z_bottom: [B, H_b, W_b] - ground truth bottom codes
        Returns:
            Dictionary with individual and total losses
        """
        outputs = self.forward(z_top, z_bottom)
        
        # Top loss
        logits_top = outputs['logits_top']  # [B, K, H_t, W_t]
        loss_top = F.cross_entropy(
            logits_top.permute(0, 2, 3, 1).reshape(-1, self.num_embeddings),
            z_top.reshape(-1)
        )
        
        # Bottom loss
        logits_bottom = outputs['logits_bottom']  # [B, K, H_b, W_b]
        loss_bottom = F.cross_entropy(
            logits_bottom.permute(0, 2, 3, 1).reshape(-1, self.num_embeddings),
            z_bottom.reshape(-1)
        )
        
        return {
            'loss_top': loss_top,
            'loss_bottom': loss_bottom,
            'total_loss': loss_top + loss_bottom
        }
    
    def sample(self,
               batch_size: int,
               top_shape: tuple,
               bottom_shape: tuple,
               device: torch.device,
               temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        """
        Hierarchical sampling: first sample top, then sample bottom conditioned on top.
        
        Args:
            batch_size: Number of samples
            top_shape: (H_t, W_t) for top level
            bottom_shape: (H_b, W_b) for bottom level
            device: Device for generation
            temperature: Sampling temperature
        
        Returns:
            (z_top, z_bottom): Tuple of sampled code indices
        """
        self.eval()
        
        # Sample top codes
        z_top = self.prior_top.sample(
            batch_size, top_shape[0], top_shape[1],
            device, temperature=temperature
        )
        
        # Condition bottom on sampled top
        z_top_emb = self.embedding_top(z_top)
        z_top_emb = z_top_emb.permute(0, 3, 1, 2).contiguous()
        z_top_up = self.upsample_top(z_top_emb)
        
        # Sample bottom codes
        z_bottom = self.prior_bottom.sample(
            batch_size, bottom_shape[0], bottom_shape[1],
            device, condition=z_top_up, temperature=temperature
        )
        
        return z_top, z_bottom
    
    def sample_with_vqvae2(self,
                           vqvae2_model,
                           batch_size: int,
                           device: torch.device,
                           temperature: float = 1.0) -> Tensor:
        """
        Generate samples by sampling codes and decoding through VQ-VAE2.
        
        Args:
            vqvae2_model: Trained VQ-VAE2 model
            batch_size: Number of images to generate
            device: Device for generation
            temperature: Sampling temperature
        
        Returns:
            generated_images: [B, C, H, W]
        """
        # Sample discrete codes
        z_top, z_bottom = self.sample(
            batch_size,
            (vqvae2_model.latent_spatial_dim_top, vqvae2_model.latent_spatial_dim_top),
            (vqvae2_model.latent_spatial_dim_bottom, vqvae2_model.latent_spatial_dim_bottom),
            device,
            temperature
        )
        
        # Convert indices to embeddings
        with torch.no_grad():
            # Top embeddings
            quant_t = vqvae2_model.vq_top.embedding(z_top.view(batch_size, -1))
            quant_t = quant_t.view(
                batch_size,
                vqvae2_model.latent_spatial_dim_top,
                vqvae2_model.latent_spatial_dim_top,
                vqvae2_model.embedding_dim
            )
            quant_t = quant_t.permute(0, 3, 1, 2).contiguous()
            
            # Bottom embeddings
            quant_b = vqvae2_model.vq_bottom.embedding(z_bottom.view(batch_size, -1))
            quant_b = quant_b.view(
                batch_size,
                vqvae2_model.latent_spatial_dim_bottom,
                vqvae2_model.latent_spatial_dim_bottom,
                vqvae2_model.embedding_dim
            )
            quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
            
            # Decode
            generated_images = vqvae2_model.decode(quant_t, quant_b)
        
        return generated_images
    
    def total_trainable_params(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

