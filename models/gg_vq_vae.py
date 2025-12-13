from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq_vae import VQVAE

# Small epsilon value for numerical stability in gradient computations
EPS = 1e-8


class GGVQVAE(VQVAE):
    """
    Gradient-Guided Vector Quantized Variational Autoencoder (GGVQ-VAE) model
    
    Inherits from VQVAE and adds gradient-guided and edge matching losses.
    """
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: Optional[List[int]] = [128, 256],
                 num_residual_layers: int = 6,
                 input_size: int = 64,
                 layer_norm: str = "none",
                 output_activation: str = "tanh",
                 recons_dist: str = "gaussian",
                 recons_reduction: str = "mean",
                 lambda_weights: Optional[List[float]] = None,
                 version: str = "v1",
                 device=None,
                 **kwargs) -> None:
        # Initialize parent VQVAE class (GGVQVAE doesn't use layer_norm, so always pass "none")
        super(GGVQVAE, self).__init__(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dims=hidden_dims,
            num_residual_layers=num_residual_layers,
            input_size=input_size,
            layer_norm="none",  # GGVQVAE always uses layer_norm="none"
            output_activation=output_activation,
            recons_dist=recons_dist,
            recons_reduction=recons_reduction,
            lambda_weights=None,  # We'll set this after adding additional objectives
            device=device,
            **kwargs
        )

        # Initialize Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0.,  2.],
                            [-1,  0.,  1.]]).unsqueeze(0).unsqueeze(0) #(1,1,3,3)

        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0) #(1,1,3,3)
        # Ensuring sobel filters can apply to RGB images - register as buffers so they move with model
        self.register_buffer('sobel_x', sobel_x.expand(3, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.expand(3, 1, 3, 3))

        # Update objectives dictionary to include gradient-guided and edge matching losses
        self.objectives = {"reconstruction_loss": self.recon_obj, "embedding_loss": None, "commitment_loss": None}
        
        if version == "v1":
            self.objectives["gradient_guided_loss"] = self.gradient_guided_loss
        elif version == "v2":
            self.objectives["gradient_guided_loss"] = self.gradient_guided_loss
            self.objectives["edge_matching_loss"] = self.edge_matching_loss_v1
        elif version == "v3":
            self.objectives["gradient_guided_loss"] = self.gradient_guided_loss
            self.objectives["edge_matching_loss"] = self.edge_matching_loss_v2
        else:
            raise ValueError(f"Version {version} not supported. Choose from: v1, v2, v3")

        # lambda_weights: dictionary matching self.objectives keys
        # Accepts either dict or list (for backward compatibility)
        if lambda_weights is None:
            # Default weights based on version
            if version == "v1":
                lambda_weights = {"reconstruction_loss": 1.0, "embedding_loss": 1.0, "commitment_loss": 0.25, "gradient_guided_loss": 1.0}
            elif version == "v2":
                lambda_weights = {"reconstruction_loss": 1.0, "embedding_loss": 1.0, "commitment_loss": 0.25, "gradient_guided_loss": 1.0, "edge_matching_loss": 1.0}
            elif version == "v3":
                lambda_weights = {"reconstruction_loss": 1.0, "embedding_loss": 1.0, "commitment_loss": 0.25, "gradient_guided_loss": 1.0, "edge_matching_loss": 1.0}
        elif isinstance(lambda_weights, list):
            # Convert list to dict based on version
            if version == "v1":
                if len(lambda_weights) != 4:
                    raise ValueError(f"GGVQVAE v1 requires 4 lambda_weights (reconstruction, embedding, commitment, gradient_guided), got {len(lambda_weights)}")
                lambda_weights = {
                    "reconstruction_loss": lambda_weights[0],
                    "embedding_loss": lambda_weights[1],
                    "commitment_loss": lambda_weights[2],
                    "gradient_guided_loss": lambda_weights[3]
                }
            elif version == "v2":
                if len(lambda_weights) != 5:
                    raise ValueError(f"GGVQVAE v2 requires 5 lambda_weights (reconstruction, embedding, commitment, gradient_guided, edge_matching), got {len(lambda_weights)}")
                lambda_weights = {
                    "reconstruction_loss": lambda_weights[0],
                    "embedding_loss": lambda_weights[1],
                    "commitment_loss": lambda_weights[2],
                    "gradient_guided_loss": lambda_weights[3],
                    "edge_matching_loss": lambda_weights[4]
                }
            elif version == "v3":
                if len(lambda_weights) != 5:
                    raise ValueError(f"GGVQVAE v3 requires 5 lambda_weights (reconstruction, embedding, commitment, gradient_guided, edge_matching), got {len(lambda_weights)}")
                lambda_weights = {
                    "reconstruction_loss": lambda_weights[0],
                    "embedding_loss": lambda_weights[1],
                    "commitment_loss": lambda_weights[2],
                    "gradient_guided_loss": lambda_weights[3],
                    "edge_matching_loss": lambda_weights[4]
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
        
        # Override lambda_weights from parent
        self.lambda_weights = lambda_weights

    def gradient_guided_loss(self, inputs, recons):
        # # Sobel applied to R,G,B
        # x_grad = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        # y_grad = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))

        # #Gradient magnitude
        # grad_mag = torch.sqrt(x_grad**2 + y_grad**2 + 1e-8)# (batch_size,C,H,W)

        # # Combining across channels - take max across channels
        # grad_max = torch.max(grad_mag, dim=1)[0]

        # # Normalizing
        # flat = grad_max.view(grad_max.size(0), -1)
        # min_val = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        # max_val = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        # grad_max = (grad_max - min_val) / (max_val - min_val + 1e-8)

        # # non-reduced reconstruction loss (B, C, H, W)
        # pixel_loss = F.mse_loss(recons, inputs, reduction='none')
        # # pixel_loss = F.binary_cross_entropy(recons, inputs, reduction='none')

        # #Gradient-guided Encoder Loss
        # loss_grad = (grad_max.unsqueeze(1) * pixel_loss).mean() #(batch_size, C, H, W) then mean across batch

        # return loss_grad
        
        # Compute gradients
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))
        
        # Edge-weighted pixel loss: BCE (pixels are in [0,1])
        grad_target = torch.sqrt(input_x**2 + input_y**2 + EPS) #(batch_size,C,H,W)
        weights = grad_target.max(dim=1)[0]  # simplified
        weights = weights / (weights.max() + EPS)  # normalize to [0,1]
        
        pixel_loss = F.binary_cross_entropy(recons, inputs, reduction='none')  # BCE here
        weighted_pixel_loss = (weights.unsqueeze(1) * pixel_loss).mean()
        
        return weighted_pixel_loss

    def edge_matching_loss_v1(self, inputs, recons):
        # Compute gradients
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))
        recon_x = F.conv2d(recons, self.sobel_x, padding=1, groups=inputs.size(1))
        recon_y = F.conv2d(recons, self.sobel_y, padding=1, groups=inputs.size(1))
        
        # Edge matching: MSE (gradients are signed, unbounded)
        edge_match_loss = F.mse_loss(recon_x, input_x) + F.mse_loss(recon_y, input_y)
        
        return edge_match_loss

    def edge_matching_loss_v2(self, inputs, recons):
        # Compute gradients
        input_x = F.conv2d(inputs, self.sobel_x, padding=1, groups=inputs.size(1))
        input_y = F.conv2d(inputs, self.sobel_y, padding=1, groups=inputs.size(1))
        recon_x = F.conv2d(recons, self.sobel_x, padding=1, groups=inputs.size(1))
        recon_y = F.conv2d(recons, self.sobel_y, padding=1, groups=inputs.size(1))
        
        # Edge matching: L1 loss (gradients are signed, unbounded)
        # Computes gradient magnitudes and compares them with L1
        # Purpose: Forces the model to reproduce edge structures
        grad_pred = torch.sqrt(recon_x**2 + recon_y**2 + EPS)
        grad_target = torch.sqrt(input_x**2 + input_y**2 + EPS)

        edge_match_loss = F.l1_loss(grad_pred, grad_target)
        
        return edge_match_loss
