
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary

from utils.objectives import mse_recon_batch_mean, mse_recon_mean
from utils.objectives import bce_with_logits_recon_batch_mean, bce_with_logits_recon_mean
from utils.objectives import laplacian_recon_batch_mean, laplacian_recon_mean
from models.vq_vae import VectorQuantizer, ResidualLayer


class VQVAE2(nn.Module):
    """
    Hierarchical Vector Quantized Variational Autoencoder (VQ-VAE-2) model.
    Uses two levels of latent variables (top and bottom).
    """
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: Optional[List[int]] = [128, 256],
                 beta: float = 0.25,
                 input_size: int = 64,
                 layer_norm: str = "none",
                 output_activation: str = "tanh",
                 recons_dist: str = "gaussian",
                 **kwargs) -> None:
        super(VQVAE2, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.input_size = input_size
        self.in_channels = in_channels
        self.beta = beta
        self._summary_mode = False
        
        # Set up reconstruction objective
        recon_obj = None
        if recons_dist == "gaussian":
            recon_obj = mse_recon_mean
            if output_activation == "tanh":
                pass  # Keep tanh
            else:
                output_activation = "tanh"  # Default to tanh for gaussian
        elif recons_dist == "bernoulli":
            recon_obj = bce_with_logits_recon_mean
            output_activation = "none"
        elif recons_dist == "laplacian":
            recon_obj = laplacian_recon_mean
            if output_activation == "tanh":
                pass  # Keep tanh
            else:
                output_activation = "tanh"  # Default to tanh for laplacian
        else:
            raise ValueError(f"Reconstruction distribution {recons_dist} not supported. Choose from: gaussian, bernoulli, laplacian")
        self.recon_obj = recon_obj

        self.objectives = {
            "reconstruction_loss": recon_obj,
            "commitment_loss": None,
            "embedding_loss": None
        }

        self.features = ["encoding_top", "encoding_bottom"]

        # Setup output activation
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "none":
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Output activation {output_activation} not supported")

        # --- Encoder (Bottom Level) ---
        # Downsamples input to bottom latent spatial dim
        modules = []
        enc_in_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(enc_in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            enc_in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(enc_in_channels, enc_in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(2): # Fewer res layers per level to keep params reasonable
            modules.append(ResidualLayer(enc_in_channels, enc_in_channels))
            
        modules.append(nn.LeakyReLU())
        
        self.enc_bottom = nn.Sequential(*modules)
        self.bottom_dim = enc_in_channels

        # --- Encoder (Top Level) ---
        # Downsamples bottom latent to top latent spatial dim
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(self.bottom_dim, out_channels=self.bottom_dim,
                          kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU())
        )
        
        modules.append(
            nn.Sequential(
                nn.Conv2d(self.bottom_dim, self.bottom_dim,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(2):
            modules.append(ResidualLayer(self.bottom_dim, self.bottom_dim))
            
        modules.append(nn.LeakyReLU())
        
        # Project to embedding dim
        modules.append(
            nn.Sequential(
                nn.Conv2d(self.bottom_dim, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )
        
        self.enc_top = nn.Sequential(*modules)

        # --- Vector Quantizers ---
        self.vq_bottom = VectorQuantizer(num_embeddings, embedding_dim, self.beta)
        self.vq_top = VectorQuantizer(num_embeddings, embedding_dim, self.beta)
        
        # Projection for bottom before VQ
        self.bottom_pre_vq = nn.Conv2d(self.bottom_dim, embedding_dim, kernel_size=1)

        # --- Decoder (Top Level) ---
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, self.bottom_dim,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )
        
        for _ in range(2):
            modules.append(ResidualLayer(self.bottom_dim, self.bottom_dim))
            
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.bottom_dim, self.bottom_dim,
                                   kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU())
        )
        
        self.dec_top = nn.Sequential(*modules)

        # --- Decoder (Bottom Level) ---
        # Takes concatenated [dec_top_output, quantized_bottom]
        modules = []
        
        # Input channels = bottom_dim (from top decoder) + embedding_dim (from bottom vq)
        dec_in_channels = self.bottom_dim + embedding_dim
        
        modules.append(
            nn.Sequential(
                nn.Conv2d(dec_in_channels, self.bottom_dim,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )
        
        for _ in range(2):
            modules.append(ResidualLayer(self.bottom_dim, self.bottom_dim))

        # Upsample back to image size
        decoder_hidden_dims = hidden_dims.copy()
        decoder_hidden_dims.reverse()
        
        # First layer takes bottom_dim
        curr_dim = self.bottom_dim
        
        for i in range(len(decoder_hidden_dims)):
            out_dim = decoder_hidden_dims[i+1] if i < len(decoder_hidden_dims)-1 else decoder_hidden_dims[i]
            
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(curr_dim,
                                       decoder_hidden_dims[i], # Output of ConvTranspose
                                       kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            curr_dim = decoder_hidden_dims[i]

        # Final layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(curr_dim, out_channels=in_channels,
                          kernel_size=3, padding=1),
                self.output_activation))
                
        self.dec_bottom = nn.Sequential(*modules)
        
        # Calculate latent dimensions for sampling
        self.latent_spatial_dim_bottom = input_size // (2 ** len(hidden_dims))
        self.latent_spatial_dim_top = self.latent_spatial_dim_bottom // 2

    def encode(self, input: Tensor) -> List[Tensor]:
        # Encode to bottom representation
        enc_b = self.enc_bottom(input)
        # Encode bottom representation to top representation
        enc_t = self.enc_top(enc_b)
        
        # Return raw latent codes (before quantization)
        # Note: We need to project enc_b to embedding dimension if we want to return it as 'code'
        # but forward() handles this differently. 
        return [enc_t, enc_b]

    def decode(self, quant_t: Tensor, quant_b: Tensor) -> Tensor:
        # Decode top
        dec_t = self.dec_top(quant_t)
        # Concatenate with bottom quantization
        dec_input = torch.cat([dec_t, quant_b], dim=1)
        # Decode bottom
        recons = self.dec_bottom(dec_input)
        return recons

    def forward(self, input: Tensor, **kwargs) -> Dict[str, Any]:
        # 1. Encoder Bottom
        enc_b = self.enc_bottom(input) # [B, bottom_dim, H/4, W/4]
        
        # 2. Encoder Top
        enc_t = self.enc_top(enc_b)    # [B, embedding_dim, H/8, W/8]
        
        # 3. VQ Top
        quant_t, diff_t, id_t = self.vq_top(enc_t)
        
        # 4. Decode Top to get conditioning for bottom
        dec_t = self.dec_top(quant_t) # [B, bottom_dim, H/4, W/4]
        
        # 5. Prepare Bottom for VQ
        # We preserve residuals: features - decoded_top
        # Or just quantize the bottom features directly. 
        # Standard VQ-VAE-2 quantizes enc_b directly.
        quant_b_input = self.bottom_pre_vq(enc_b) # [B, embedding_dim, H/4, W/4]
        
        # 6. VQ Bottom
        quant_b, diff_b, id_b = self.vq_bottom(quant_b_input)
        
        # 7. Decoder Bottom (with conditioning from Top)
        # Concatenate along channel dimension
        dec_input = torch.cat([dec_t, quant_b], dim=1)
        recons = self.dec_bottom(dec_input)

        # Sum losses
        commitment_loss = diff_t + diff_b
        embedding_loss = id_t + id_b # In existing implementation this variable holds embedding loss
        
        if isinstance(embedding_loss, tuple): # Handle if vq returns tuple vs tensor
             # This part depends on exact return of VectorQuantizer.forward
             # Based on models/vq_vae.py: returns (quantized, commitment_loss, embedding_loss)
             pass

        outputs = {
            "recons": recons,
            "quantized_top": quant_t,
            "quantized_bottom": quant_b,
            "commitment_loss": commitment_loss,
            "embedding_loss": embedding_loss,
        }
        
        if getattr(self, "_summary_mode", False):
            return outputs["recons"]
            
        return outputs

    def loss_function(self, inputs, args: dict) -> dict:
        recons = args["recons"]
        commitment_loss = args["commitment_loss"]
        embedding_loss = self.beta * args["embedding_loss"]
        recon_loss = self.recon_obj(recons, inputs)
        
        return {
            "reconstruction_loss": recon_loss,
            "commitment_loss": commitment_loss,
            "embedding_loss": embedding_loss,
        }

    def sample(self, num_samples=1, device=None):
        """
        Sample uniformly from codebooks (random noise generation).
        Requires a prior (e.g. PixelCNN) for meaningful generation.
        """
        self.eval()
        with torch.no_grad():
            # Random top indices
            rand_ind_t = torch.randint(
                0, self.num_embeddings,
                (num_samples, self.latent_spatial_dim_top * self.latent_spatial_dim_top),
                device=device
            )
            quant_t = self.vq_top.embedding(rand_ind_t)
            quant_t = quant_t.view(num_samples, self.latent_spatial_dim_top, 
                                   self.latent_spatial_dim_top, self.embedding_dim)
            quant_t = quant_t.permute(0, 3, 1, 2).contiguous()
            
            # Random bottom indices
            rand_ind_b = torch.randint(
                0, self.num_embeddings,
                (num_samples, self.latent_spatial_dim_bottom * self.latent_spatial_dim_bottom),
                device=device
            )
            quant_b = self.vq_bottom.embedding(rand_ind_b)
            quant_b = quant_b.view(num_samples, self.latent_spatial_dim_bottom,
                                   self.latent_spatial_dim_bottom, self.embedding_dim)
            quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
            
            generated_samples = self.decode(quant_t, quant_b)
            
        return generated_samples

    def total_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_summary(self):
        was_training = self.training
        try:
            self._summary_mode = True
            self.vq_top._summary_mode = True
            self.vq_bottom._summary_mode = True
            self.train(False)
            param_device = next(self.parameters()).device
            summary_device = "cuda" if param_device.type == "cuda" else "cpu"
            result = summary(
                self,
                (self.in_channels, self.input_size, self.input_size),
                device=summary_device,
            )
            return result
        except Exception as e:
            print(f"Error printing model summary: {e}")
            return None
        finally:
            self._summary_mode = False
            self.vq_top._summary_mode = False
            self.vq_bottom._summary_mode = False
            self.train(was_training)

