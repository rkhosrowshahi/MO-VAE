"""
Image quality metrics for evaluation: SSIM, SSNR, and FID
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy import linalg


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    SSIM measures the similarity between two images based on luminance, contrast,
    and structure. It ranges from -1 to 1, with 1 indicating identical images.
    Higher values indicate better reconstruction quality.
    
    Args:
        img1: First image tensor of shape (B, C, H, W) in range [0, 1] or [-1, 1]
        img2: Second image tensor of shape (B, C, H, W) in range [0, 1] or [-1, 1]
        window_size: Size of the Gaussian window for local statistics computation.
                    Must be odd. Default is 11.
        size_average: If True, return scalar average SSIM across entire batch.
                     If False, return per-image SSIM values of shape (B,).
        
    Returns:
        torch.Tensor: SSIM value(s). If size_average=True, returns scalar tensor.
                     If size_average=False, returns tensor of shape (B,).
    """
    # Normalize to [0, 1] if in [-1, 1] range
    if img1.min() < 0:
        img1 = (img1 + 1) / 2
    if img2.min() < 0:
        img2 = (img2 + 1) / 2
    
    # Clamp to valid range
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    channels = img1.size(1)
    window = create_window(window_size, img1.device, img1.dtype, channels=channels)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def create_window(window_size, device, dtype, channels=3):
    """
    Create a 2D Gaussian window for SSIM computation.
    
    Generates a normalized 2D Gaussian kernel that is used to compute local
    statistics (mean, variance, covariance) in the SSIM calculation.
    
    Args:
        window_size: Size of the square window (must be odd)
        device: Device to create the window on (e.g., 'cuda:0' or 'cpu')
        dtype: Data type for the window tensor
        channels: Number of channels (window is replicated for each channel)
        
    Returns:
        torch.Tensor: Gaussian window of shape (channels, 1, window_size, window_size)
    """
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channels, 1, window_size, window_size).contiguous().to(device).to(dtype)
    return window


def ssnr(img1, img2):
    """
    Compute Signal-to-Noise Ratio (SSNR) between original and reconstructed images.
    
    SSNR measures the ratio of signal power (variance of original image) to noise
    power (MSE between original and reconstructed). Higher values indicate better
    reconstruction quality. The result is expressed in decibels (dB).
    
    Args:
        img1: Original image tensor of shape (B, C, H, W) in range [0, 1] or [-1, 1]
        img2: Reconstructed image tensor of shape (B, C, H, W) in range [0, 1] or [-1, 1]
        
    Returns:
        float: Average SSNR value in dB across the batch. Higher is better.
    """
    # Normalize to [0, 1] if in [-1, 1] range
    if img1.min() < 0:
        img1 = (img1 + 1) / 2
    if img2.min() < 0:
        img2 = (img2 + 1) / 2
    
    # Clamp to valid range
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Compute signal power (variance of original image)
    signal_power = torch.var(img1, dim=[1, 2, 3])  # (B,)
    
    # Compute noise power (MSE between original and reconstructed)
    noise = img1 - img2
    noise_power = torch.mean(noise ** 2, dim=[1, 2, 3])  # (B,)
    
    # Avoid division by zero
    noise_power = torch.clamp(noise_power, min=1e-10)
    
    # Compute SSNR in dB
    snr = signal_power / noise_power
    snr_db = 10 * torch.log10(snr)
    
    return snr_db.mean().item()


class InceptionV3(nn.Module):
    """
    Pretrained InceptionV3 network for FID computation.
    
    This wrapper extracts features from the penultimate layer of a pretrained
    InceptionV3 network. The features are used to compute the Fréchet Inception
    Distance (FID) between real and generated images.
    """
    def __init__(self, device='cuda'):
        """
        Initialize the InceptionV3 feature extractor.
        
        Args:
            device: Device to load the model on (e.g., 'cuda:0' or 'cpu')
        """
        super(InceptionV3, self).__init__()
        self.device = device
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        inception.fc = nn.Identity()
        inception.eval()
        self.model = inception.to(device)
        
    def forward(self, x):
        """
        Extract features from images using InceptionV3.
        
        Images are resized to 299x299 if needed, normalized to ImageNet statistics,
        and passed through the InceptionV3 network to extract 2048-dimensional
        feature vectors.
        
        Args:
            x: Input image tensor of shape (B, C, H, W) in range [0, 1] or [-1, 1]
               Images will be automatically resized to 299x299 if needed.
        
        Returns:
            torch.Tensor: Feature vectors of shape (B, 2048)
        """
        # Resize to 299x299 if needed
        if x.size(-1) != 299 or x.size(-2) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1] if in [-1, 1] range
        if x.min() < 0:
            x = (x + 1) / 2
        
        # Clamp to valid range
        x = torch.clamp(x, 0, 1)
        
        # Normalize for Inception (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # Get features from the last pooling layer
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


def calculate_fid(real_images, fake_images, device='cuda', batch_size=50):
    """
    Calculate Fréchet Inception Distance (FID) between real and fake images.
    
    FID measures the distance between the distribution of real and generated images
    in the feature space of a pretrained Inception network. It computes the
    Fréchet distance between two multivariate Gaussians fitted to the feature
    representations.
    
    Note: FID is a distance/error metric where LOWER is better. Lower FID indicates
    that the generated images are more similar to real images in terms of their
    distribution in the Inception feature space.
    
    Args:
        real_images: Tensor of real images of shape (N, C, H, W) in range [0, 1] or [-1, 1]
        fake_images: Tensor of fake/reconstructed images of shape (M, C, H, W) in range [0, 1] or [-1, 1]
        device: Device to run computation on (e.g., 'cuda:0' or 'cpu')
        batch_size: Batch size for feature extraction. Larger batches use more memory
                   but may be faster. Default is 50.
        
    Returns:
        float: FID distance. Lower is better, with 0 being the theoretical lower bound
               (achieved when distributions are identical).
    """
    inception = InceptionV3(device=device)
    inception.eval()
    
    def get_features(images):
        features_list = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(device)
                features = inception(batch)
                features_list.append(features.cpu())
        return torch.cat(features_list, dim=0).numpy()
    
    real_features = get_features(real_images)
    fake_features = get_features(fake_images)
    
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

