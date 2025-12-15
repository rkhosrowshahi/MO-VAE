"""
Image quality metrics for evaluation: SSIM, SSNR, FID, IS, and Precision/Recall
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist


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
    
    # Avoid division by zero and log(0)
    # Clamp both to avoid inf/-inf in edge cases (constant images or perfect reconstruction)
    signal_power = torch.clamp(signal_power, min=1e-10)
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
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
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


class InceptionV3ForIS(nn.Module):
    """
    Pretrained InceptionV3 network for Inception Score (IS) computation.
    
    This wrapper extracts class predictions (logits) from a pretrained InceptionV3
    network. The predictions are used to compute the Inception Score, which measures
    both quality and diversity of generated images.
    """
    def __init__(self, device='cuda'):
        """
        Initialize the InceptionV3 classifier.
        
        Args:
            device: Device to load the model on (e.g., 'cuda:0' or 'cpu')
        """
        super(InceptionV3ForIS, self).__init__()
        self.device = device
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        inception.eval()
        self.model = inception.to(device)
        
    def forward(self, x):
        """
        Get class predictions from images using InceptionV3.
        
        Images are resized to 299x299 if needed, normalized to ImageNet statistics,
        and passed through the InceptionV3 network to get class logits.
        
        Args:
            x: Input image tensor of shape (B, C, H, W) in range [0, 1] or [-1, 1]
               Images will be automatically resized to 299x299 if needed.
        
        Returns:
            torch.Tensor: Class logits of shape (B, 1000)
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
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # Get class predictions
        x = self.model(x)
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
    
    # Handle edge case where covariance is scalar (single sample)
    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[sigma2]])
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Calculate sqrt of product between cov
    # Product might be almost singular, use offset for numerical stability
    offset = np.eye(sigma1.shape[0]) * 1e-6
    covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    
    # Check and correct imaginary numbers from sqrt
    # (small imaginary components are numerical artifacts)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Check for NaN (can occur with degenerate covariances)
    if np.isnan(covmean).any():
        # Fall back: return FID without the covariance term
        return float(ssdiff + np.trace(sigma1) + np.trace(sigma2))
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return float(fid)


def calculate_inception_score(images, device='cuda', batch_size=50, splits=10):
    """
    Calculate Inception Score (IS) for generated images.
    
    Inception Score measures both the quality and diversity of generated images.
    It computes exp(E[KL(p(y|x) || p(y))]) where:
    - p(y|x) is the predicted class distribution for each image
    - p(y) is the marginal class distribution across all images
    
    Higher IS indicates better quality (sharp, recognizable images) and diversity
    (different images belong to different classes). Typical values range from 1 to
    around 10-15 for good generative models.
    
    Args:
        images: Tensor of generated images of shape (N, C, H, W) in range [0, 1] or [-1, 1]
        device: Device to run computation on (e.g., 'cuda:0' or 'cpu')
        batch_size: Batch size for prediction extraction. Larger batches use more memory
                   but may be faster. Default is 50.
        splits: Number of splits to compute IS over for more stable estimates.
               The final IS is the mean across splits. Default is 10.
        
    Returns:
        tuple: (mean_is, std_is) where mean_is is the average IS across splits and
               std_is is the standard deviation. Higher is better.
    """
    inception = InceptionV3ForIS(device=device)
    inception.eval()
    
    # Get predictions for all images
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            logits = inception(batch)
            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    num_images = preds.shape[0]
    
    # Add small epsilon for numerical stability to avoid log(0)
    eps = 1e-16
    preds = np.clip(preds, eps, 1.0)
    
    # Compute IS for each split
    scores = []
    for i in range(splits):
        part = preds[i * (num_images // splits): (i + 1) * (num_images // splits)]
        if len(part) == 0:
            continue
        py = np.mean(part, axis=0)  # Marginal distribution p(y)
        py = np.clip(py, eps, 1.0)  # Ensure no zeros in marginal
        # KL divergence: sum(p(y|x) * log(p(y|x) / p(y)))
        kl_divs = np.sum(part * (np.log(part) - np.log(py)), axis=1)
        scores.append(np.exp(np.mean(kl_divs)))
    
    return np.mean(scores), np.std(scores)


def calculate_precision_recall(real_images, fake_images, device='cuda', batch_size=50, k=5):
    """
    Calculate Precision and Recall for generated images using k-nearest neighbors.
    
    Precision and Recall measure the quality and coverage of generated images:
    - Precision: Fraction of generated images that are realistic (within the real
                 distribution). Measures quality - higher precision means more
                 generated images are realistic.
    - Recall: Fraction of real images that are covered by the generated distribution.
              Measures diversity/coverage - higher recall means the generated
              distribution better covers the real distribution.
    
    The metrics are computed in the feature space of a pretrained Inception network
    using k-nearest neighbors. This follows the method from "Assessing Generative
    Models via Precision and Recall" (Kynkaanniemi et al., 2019).
    
    Args:
        real_images: Tensor of real images of shape (N, C, H, W) in range [0, 1] or [-1, 1]
        fake_images: Tensor of fake/generated images of shape (M, C, H, W) in range [0, 1] or [-1, 1]
        device: Device to run computation on (e.g., 'cuda:0' or 'cpu')
        batch_size: Batch size for feature extraction. Larger batches use more memory
                   but may be faster. Default is 50.
        k: Number of nearest neighbors to use for manifold estimation. Default is 5.
        
    Returns:
        tuple: (precision, recall) where both are floats in [0, 1]. Higher is better
               for both metrics.
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
    
    # Compute pairwise distances
    # Distance from each fake image to all real images
    distances_fake_to_real = cdist(fake_features, real_features, metric='euclidean')
    # Distance from each real image to all fake images
    distances_real_to_fake = cdist(real_features, fake_features, metric='euclidean')
    # Distance from each real image to all real images (for k-NN in real manifold)
    distances_real_to_real = cdist(real_features, real_features, metric='euclidean')
    # Distance from each fake image to all fake images (for k-NN in fake manifold)
    distances_fake_to_fake = cdist(fake_features, fake_features, metric='euclidean')
    
    # Set diagonal to large value to exclude self-distances
    np.fill_diagonal(distances_real_to_real, np.inf)
    np.fill_diagonal(distances_fake_to_fake, np.inf)
    
    # Compute k-NN radius for each real sample (distance to k-th nearest real neighbor)
    # Using k-1 index because partition returns 0-indexed positions
    real_knn_radii = np.partition(distances_real_to_real, k-1, axis=1)[:, k-1]
    
    # Compute k-NN radius for each fake sample (distance to k-th nearest fake neighbor)
    fake_knn_radii = np.partition(distances_fake_to_fake, k-1, axis=1)[:, k-1]
    
    # Precision: For each fake sample, find its NEAREST real neighbor,
    # then check if the distance is within that neighbor's k-NN radius
    # (Algorithm 1 from Kynkaanniemi et al. 2019)
    precision_scores = []
    for i in range(len(fake_features)):
        # Find nearest real neighbor
        nearest_real_idx = np.argmin(distances_fake_to_real[i])
        dist_to_nearest = distances_fake_to_real[i, nearest_real_idx]
        # Check if within the k-NN radius of that nearest real sample
        in_manifold = dist_to_nearest <= real_knn_radii[nearest_real_idx]
        precision_scores.append(float(in_manifold))
    
    precision = np.mean(precision_scores)
    
    # Recall: For each real sample, find its NEAREST fake neighbor,
    # then check if the distance is within that neighbor's k-NN radius
    recall_scores = []
    for i in range(len(real_features)):
        # Find nearest fake neighbor
        nearest_fake_idx = np.argmin(distances_real_to_fake[i])
        dist_to_nearest = distances_real_to_fake[i, nearest_fake_idx]
        # Check if within the k-NN radius of that nearest fake sample
        in_manifold = dist_to_nearest <= fake_knn_radii[nearest_fake_idx]
        recall_scores.append(float(in_manifold))
    
    recall = np.mean(recall_scores)
    
    return precision, recall

