from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .vq_vae2 import VQVAE2

# Small epsilon value for numerical stability in gradient computations
EPS = 1e-8


class GGVQVAE2(VQVAE2):
    """
    Gradient-Guided Vector Quantized Variational Autoencoder 2 (GG-VQ-VAE2).
    
    Inherits from VQVAE2 (hierarchical two-level VQ-VAE) and adds gradient-guided
    and edge matching losses from GG-VQ-VAE-V3.
    """
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: Optional[List[int]] = [128, 256],
                 num_residual_layers: int = 2,
                 input_size: int = 64,
                 layer_norm: str = "none",
                 recons_activation: str = "tanh",
                 recons_objective: str = "mse",
                 lambda_weights: Optional[Dict[str, float]] = None,
                 version: str = "v3",
                 device=None,
                 **kwargs) -> None:
        # Build base VQVAE2 objectives first - we'll override after super().__init__
        super().__init__(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dims=hidden_dims,
            num_residual_layers=num_residual_layers,
            input_size=input_size,
            layer_norm=layer_norm,
            recons_activation=recons_activation,
            recons_objective=recons_objective,
            lambda_weights=None,  # Set after adding GG objectives
            device=device,
            **kwargs
        )

        # Initialize Sobel filters for gradient computation (same as GGVQVAE)
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x.expand(in_channels, 1, 3, 3).clone())
        self.register_buffer('sobel_y', sobel_y.expand(in_channels, 1, 3, 3).clone())

        # Add GG-V3 objectives (gradient_guided_loss, edge_matching_loss)
        self.objectives["gradient_guided_loss"] = self.gradient_guided_loss
        self.objectives["edge_matching_loss"] = self.edge_matching_loss_v2

        # lambda_weights for GG-VQ-VAE2: 5 objectives
        if lambda_weights is None:
            lambda_weights = {
                "reconstruction_loss": 1.0,
                "commitment_loss": 1.0,
                "embedding_loss": 0.25,
                "gradient_guided_loss": 1.0,
                "edge_matching_loss": 1.0,
            }
        elif isinstance(lambda_weights, list):
            if len(lambda_weights) != 5:
                raise ValueError(
                    f"GGVQVAE2 requires 5 lambda_weights "
                    f"(reconstruction, commitment, embedding, gradient_guided, edge_matching), "
                    f"got {len(lambda_weights)}"
                )
            lambda_weights = {
                "reconstruction_loss": lambda_weights[0],
                "commitment_loss": lambda_weights[1],
                "embedding_loss": lambda_weights[2],
                "gradient_guided_loss": lambda_weights[3],
                "edge_matching_loss": lambda_weights[4],
            }
        elif isinstance(lambda_weights, dict):
            expected_keys = set(self.objectives.keys())
            provided_keys = set(lambda_weights.keys())
            if expected_keys != provided_keys:
                missing = expected_keys - provided_keys
                extra = provided_keys - expected_keys
                error_msg = "lambda_weights keys must match objectives keys. "
                if missing:
                    error_msg += f"Missing: {missing}. "
                if extra:
                    error_msg += f"Extra: {extra}."
                raise ValueError(error_msg)
        else:
            raise TypeError(f"lambda_weights must be dict or list, got {type(lambda_weights)}")

        self.lambda_weights = lambda_weights

    def gradient_guided_loss(self, inputs: Tensor, recons: Tensor) -> Tensor:
        """Edge-weighted pixel loss (BCE) - from GG-VQ-VAE-V3. Use recons_activation='sigmoid' for [0,1] range."""
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))

        grad_target = torch.sqrt(input_x ** 2 + input_y ** 2 + EPS)
        weights = grad_target.max(dim=1)[0]
        weights = weights / (weights.max() + EPS)

        pixel_loss = F.mse_loss(recons, inputs, reduction='none')
        weighted_pixel_loss = (weights.unsqueeze(1) * pixel_loss).mean()

        return weighted_pixel_loss

    def edge_matching_loss_v2(self, inputs: Tensor, recons: Tensor) -> Tensor:
        """L1 loss on gradient magnitudes - from GG-VQ-VAE-V3."""
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))
        recon_x = F.conv2d(recons, self.sobel_x, padding=1, groups=inputs.size(1))
        recon_y = F.conv2d(recons, self.sobel_y, padding=1, groups=inputs.size(1))

        grad_pred = torch.sqrt(recon_x ** 2 + recon_y ** 2 + EPS)
        grad_target = torch.sqrt(input_x ** 2 + input_y ** 2 + EPS)

        return F.smooth_l1_loss(grad_pred, grad_target)

    def loss_function(self, inputs: Tensor, args: Dict[str, Any]) -> Dict[str, Tensor]:
        recons = args["recons"]
        commitment_loss = args["commitment_loss"]
        embedding_loss = args["embedding_loss"]

        recon_loss = self.recon_obj(inputs, recons)
        grad_guided_loss = self.gradient_guided_loss(inputs, recons)
        edge_match_loss = self.edge_matching_loss_v2(inputs, recons)

        weighted_recon_loss = self.lambda_weights["reconstruction_loss"] * recon_loss
        weighted_commitment_loss = self.lambda_weights["commitment_loss"] * commitment_loss
        weighted_embedding_loss = self.lambda_weights["embedding_loss"] * embedding_loss
        weighted_grad_guided_loss = self.lambda_weights["gradient_guided_loss"] * grad_guided_loss
        weighted_edge_match_loss = self.lambda_weights["edge_matching_loss"] * edge_match_loss

        total_loss = (
            weighted_recon_loss
            + weighted_commitment_loss
            + weighted_embedding_loss
            + weighted_grad_guided_loss
            + weighted_edge_match_loss
        )

        return {
            "reconstruction_loss": weighted_recon_loss,
            "commitment_loss": weighted_commitment_loss,
            "embedding_loss": weighted_embedding_loss,
            "gradient_guided_loss": weighted_grad_guided_loss,
            "edge_matching_loss": weighted_edge_match_loss,
            "total_loss": total_loss,
        }

    def print_model_summary(self):
        """Override to fix parent's attribute names (quantize_t, quantize_b)."""
        self.to(self.device)

        was_training = self.training
        model_device = (
            next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else torch.device('cpu')
        )

        if model_device.type == 'cuda':
            summary_device = "cuda"
            if model_device.index is not None:
                original_device = torch.cuda.current_device()
                torch.cuda.set_device(model_device.index)
        else:
            summary_device = "cpu"
            original_device = None

        from torchsummary import summary
        try:
            self._summary_mode = True
            self.quantize_t._summary_mode = True
            self.quantize_b._summary_mode = True
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
            if (
                model_device.type == 'cuda'
                and model_device.index is not None
                and original_device is not None
            ):
                torch.cuda.set_device(original_device)
            self._summary_mode = False
            self.quantize_t._summary_mode = False
            self.quantize_b._summary_mode = False
            self.train(was_training)
