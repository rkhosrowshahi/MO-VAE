
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary

from utils.objectives import mse_per_image_sum, mse_per_pixel_mean, mse_total_batch_sum_scaled
from utils.objectives import bce_with_logits_per_image_sum, bce_with_logits_per_pixel_mean
from utils.objectives import laplacian_per_image_sum, laplacian_per_pixel_mean
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
                 input_size: int = 64,
                 layer_norm: str = "none",
                 output_activation: str = "tanh",
                 recons_dist: str = "gaussian",
                 recons_reduction: str = "mean",
                 lambda_weights: Optional[List[float]] = None,
                 device=None,
                 **kwargs) -> None:
        super(VQVAE2, self).__init__()

        self.device = device
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.input_size = input_size
        self.in_channels = in_channels
        self._summary_mode = False
        
        # Set up reconstruction objective
        recon_obj = None
        if recons_dist == "gaussian":
            if recons_reduction == "mean":
                recon_obj = mse_per_pixel_mean
            elif recons_reduction == "sum":
                recon_obj = mse_per_image_sum
            elif recons_reduction == "scaled_sum":
                recon_obj = mse_total_batch_sum_scaled
            else:
                raise ValueError(f"MSE reduction {recons_reduction} not supported. Choose from: mean, sum, scaled_sum")
            if output_activation == "tanh":
                pass  # Keep tanh
            else:
                output_activation = "tanh"  # Default to tanh for gaussian
        elif recons_dist == "bernoulli":
            if recons_reduction == "mean":
                recon_obj = bce_with_logits_per_pixel_mean
            elif recons_reduction == "sum":
                recon_obj = bce_with_logits_per_image_sum
            else:
                 raise ValueError(f"BCE reduction {recons_reduction} not supported. Choose from: mean, sum")
            output_activation = "none"
        elif recons_dist == "laplacian":
            if recons_reduction == "mean":
                recon_obj = laplacian_per_pixel_mean
            elif recons_reduction == "sum":
                recon_obj = laplacian_per_image_sum
            else:
                 raise ValueError(f"Laplacian reduction {recons_reduction} not supported. Choose from: mean, sum")
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

        # lambda_weights: dictionary matching self.objectives keys
        # Accepts either dict or list (for backward compatibility)
        if lambda_weights is None:
            lambda_weights = {"reconstruction_loss": 1.0, "commitment_loss": 1.0, "embedding_loss": 1.0}
        elif isinstance(lambda_weights, list):
            # Convert list to dict: [reconstruction_weight, commitment_weight, embedding_weight]
            if len(lambda_weights) != 3:
                raise ValueError(f"VQVAE2 requires 3 lambda_weights (reconstruction, commitment, embedding), got {len(lambda_weights)}")
            lambda_weights = {
                "reconstruction_loss": lambda_weights[0],
                "commitment_loss": lambda_weights[1],
                "embedding_loss": lambda_weights[2]
            }
        elif isinstance(lambda_weights, dict):
            # Validate dict keys match objectives
            expected_keys = set(self.objectives.keys())
            provided_keys = set(lambda_weights.keys())
            if expected_keys != provided_keys:
                missing = expected_keys - provided_keys
                extra = provided_keys - expected_keys
                error_msg = f"lambda_weights keys must match objectives keys. "
                if missing:
                    error_msg += f"Missing: {missing}. "
                if extra:
                    error_msg += f"Extra: {extra}."
                raise ValueError(error_msg)
        else:
            raise TypeError(f"lambda_weights must be dict or list, got {type(lambda_weights)}")
        
        self.lambda_weights = lambda_weights

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
        self.vq_bottom = VectorQuantizer(num_embeddings, embedding_dim)
        self.vq_top = VectorQuantizer(num_embeddings, embedding_dim)
        
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
        embedding_loss = args["embedding_loss"]
        recon_loss = self.recon_obj(inputs, recons)
        
        # Apply lambda_weights using dictionary keys matching self.objectives
        weighted_recon_loss = self.lambda_weights["reconstruction_loss"] * recon_loss
        weighted_commitment_loss = self.lambda_weights["commitment_loss"] * commitment_loss
        weighted_embedding_loss = self.lambda_weights["embedding_loss"] * embedding_loss
        
        return {
            "reconstruction_loss": weighted_recon_loss,
            "commitment_loss": weighted_commitment_loss,
            "embedding_loss": weighted_embedding_loss,
        }

    def get_code_indices(self, input: Tensor) -> Dict[str, Tensor]:
        """
        Extract discrete code indices from input.
        Used for training PixelCNN prior.
        
        Args:
            input: [B, C, H, W] input images
        
        Returns:
            Dictionary with 'indices_top' [B, H_t, W_t] and 'indices_bottom' [B, H_b, W_b]
        """
        self.eval()
        with torch.no_grad():
            # Encode
            enc_b = self.enc_bottom(input)
            enc_t = self.enc_top(enc_b)
            
            # Get top indices
            enc_t_perm = enc_t.permute(0, 2, 3, 1).contiguous()
            flat_enc_t = enc_t_perm.view(-1, self.embedding_dim)
            
            # Compute distances to top codebook
            dist_t = torch.sum(flat_enc_t ** 2, dim=1, keepdim=True) + \
                     torch.sum(self.vq_top.embedding.weight ** 2, dim=1) - \
                     2 * torch.matmul(flat_enc_t, self.vq_top.embedding.weight.t())
            indices_t = torch.argmin(dist_t, dim=1)
            
            # Reshape to spatial dimensions
            B = input.size(0)
            indices_t = indices_t.view(B, self.latent_spatial_dim_top, self.latent_spatial_dim_top)
            
            # Get bottom indices
            quant_b_input = self.bottom_pre_vq(enc_b)
            quant_b_input_perm = quant_b_input.permute(0, 2, 3, 1).contiguous()
            flat_enc_b = quant_b_input_perm.view(-1, self.embedding_dim)
            
            # Compute distances to bottom codebook
            dist_b = torch.sum(flat_enc_b ** 2, dim=1, keepdim=True) + \
                     torch.sum(self.vq_bottom.embedding.weight ** 2, dim=1) - \
                     2 * torch.matmul(flat_enc_b, self.vq_bottom.embedding.weight.t())
            indices_b = torch.argmin(dist_b, dim=1)
            
            # Reshape to spatial dimensions
            indices_b = indices_b.view(B, self.latent_spatial_dim_bottom, self.latent_spatial_dim_bottom)
            
        return {
            'indices_top': indices_t,
            'indices_bottom': indices_b
        }

    def sample(self, num_samples=1, device=None):
        """
        Sample uniformly from codebooks (random noise generation).
        Requires a prior (e.g. PixelCNN) for meaningful generation.
        
        For proper sampling with learned prior, use:
            from models.pixelcnn_prior import HierarchicalPixelCNN
            prior = HierarchicalPixelCNN(...)
            samples = prior.sample_with_vqvae2(vqvae2_model, batch_size, device)
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
        # Get the device the model is actually on (from its parameters)
        model_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        
        # Determine summary device (torchsummary only accepts 'cuda' or 'cpu')
        if model_device.type == 'cuda':
            summary_device = "cuda"
            # If model is on a specific CUDA device (e.g., cuda:7), temporarily set it as default
            # so torchsummary creates inputs on the correct device
            if model_device.index is not None:
                original_device = torch.cuda.current_device()
                torch.cuda.set_device(model_device.index)
        else:
            summary_device = "cpu"
            original_device = None
        
        try:
            self._summary_mode = True
            self.vq_top._summary_mode = True
            self.vq_bottom._summary_mode = True
            self.train(False)
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
            # Restore original CUDA device if we changed it
            if model_device.type == 'cuda' and model_device.index is not None and original_device is not None:
                torch.cuda.set_device(original_device)
            self._summary_mode = False
            self.vq_top._summary_mode = False
            self.vq_bottom._summary_mode = False
            self.train(was_training)

