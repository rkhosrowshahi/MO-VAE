import os
import numpy as np
import torch
from torchvision import datasets, transforms
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset


def _celeba_data_exists(data_dir):
    """Return True if CelebA images and split file exist under data_dir (root for torchvision CelebA)."""
    celeba_dir = os.path.join(data_dir, "celeba")
    img_dir = os.path.join(celeba_dir, "img_align_celeba")
    partition_file = os.path.join(celeba_dir, "list_eval_partition.txt")
    if not os.path.isdir(img_dir):
        return False
    if not os.path.isfile(partition_file):
        return False
    # At least one image
    try:
        names = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        return len(names) > 0
    except OSError:
        return False


def _ensure_celeba_downloaded(data_dir):
    """If CelebA is not present at data_dir, download it via torchvision (requires gdown)."""
    if _celeba_data_exists(data_dir):
        return
    os.makedirs(data_dir, exist_ok=True)
    print(f"CelebA not found at {data_dir}. Downloading CelebA (this may take a while)...")
    try:
        # Trigger torchvision download: creates data_dir/celeba/ and downloads img_align_celeba etc.
        datasets.CelebA(root=data_dir, split="train", download=True)
        if not _celeba_data_exists(data_dir):
            raise RuntimeError("CelebA download completed but data directory is still missing or invalid.")
        print("CelebA download finished successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download CelebA to {data_dir}. "
            "You can manually download from https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8 and "
            "extract so that 'celeba/img_align_celeba' and 'celeba/list_eval_partition.txt' exist under the data_dir."
        ) from e

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Sets the random seed for NumPy, PyTorch CPU operations, and all CUDA devices
    to ensure reproducible results across runs.
    
    Args:
        seed: Integer seed value to use for all random number generators
        
    Returns:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    
    Useful for tracking metrics during training/evaluation, such as loss values
    or accuracy scores. Maintains running statistics that can be updated incrementally.
    """
    def __init__(self):
        """Initialize the meter with zero values."""
        self.reset()
        
    def reset(self):
        """
        Reset all statistics to zero.
        
        Returns:
            None
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        """
        Update statistics with a new value.
        
        Args:
            val: New value to add to the running average
            n: Number of samples this value represents (default: 1)
               Useful when val is already an average over n samples
        
        Returns:
            None
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        """
        String representation of the average value.
        
        Returns:
            str: Formatted average value with 4 decimal places
        """
        return f"{self.avg:.4f}"

class MyCelebA(datasets.CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True

class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset      # already has .with_transform() applied or not
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]                # this is a dict: {"image": PIL, "label": int}
        
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        label = item["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label          # ← classic (tensor, label) tuple!

def get_dataset(dataset_name, data_dir='./data', normalize=False):
    """
    Load and prepare a dataset for training/evaluation.
    
    Supports CIFAR-10, CIFAR-100, ImageNet, CelebA, Oxford Flowers-102, and AFHQ (animal faces) datasets.
    Applies appropriate transforms including data augmentation for training sets and optional normalization.
    
    Args:
        dataset_name: Name of the dataset. Supported values:
                     - 'CIFAR10': CIFAR-10 dataset (32x32 images, 10 classes)
                     - 'CIFAR100': CIFAR-100 dataset (32x32 images, 100 classes)
                     - 'ImageNet': ImageNet dataset (224x224 images, 1000 classes)
                     - 'CelebA': CelebA face dataset (64x64 images)
                     - 'Oxford-Flower-102': Oxford Flower 102 (train+test as train, validation as test)
                     - 'animal-face' or 'afhq': AFHQ animal faces (train only; same data used for train and test)
        data_dir: Root directory where datasets are stored or will be downloaded.
                 Default is './data'.
        normalize: If True, applies normalization transforms using dataset-specific
                  mean and std values. Default is False.
        
    Returns:
        tuple: (train_dataset, test_dataset, input_size)
            - train_dataset: PyTorch Dataset object for training
            - test_dataset: PyTorch Dataset object for testing/validation
            - input_size: Integer size of input images (e.g., 32 for CIFAR, 224 for ImageNet)
            
    Raises:
        ValueError: If dataset_name is not one of the supported datasets
        
    Note:
        - CIFAR-10/100 and ImageNet will be automatically downloaded if not present
        - CelebA requires manual download (see MyCelebA class docstring)
        - Training sets include random horizontal flip augmentation
        - All datasets are converted to tensors with values in [0, 1]
        - If normalize=True, values are normalized to approximately [-1, 1] range
    """
    if dataset_name.lower() == 'cifar10':
        # CIFAR-10 normalization values
        # Mean and std calculated from CIFAR-10 training set
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        input_size = 32
        
        transform_train = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        
        transform_test = [transforms.ToTensor()]
        
        if normalize:
            print("Normalizing CIFAR10 train and test datasets")
            transform_train.append(transforms.Normalize(mean, std))
            transform_test.append(transforms.Normalize(mean, std))

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    elif dataset_name.lower() == 'cifar100':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        input_size = 32
        
        transform_train = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        
        transform_test = [transforms.ToTensor()]

        if normalize:
            print("Normalizing CIFAR100 train and test datasets")
            transform_train.append(transforms.Normalize(mean, std))
            transform_test.append(transforms.Normalize(mean, std))

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    elif dataset_name.lower() == 'imagenet':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        input_size = 256

        train_dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="train")
        test_dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="test")

        train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        
        test_transforms = [
            transforms.ToTensor()
        ]

        if normalize:
            train_transforms.append(transforms.Normalize(mean, std))
            test_transforms.append(transforms.Normalize(mean, std))

        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)

        train_dataset = HFImageDataset(train_dataset, transform=train_transforms)
        test_dataset  = HFImageDataset(test_dataset,  transform=test_transforms)
    elif dataset_name.lower() == "celeba":
        input_size = 64
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        train_transforms = [
            transforms.CenterCrop(148),
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ]
        
        val_transforms = [
            transforms.CenterCrop(148),
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ]

        if normalize:
            train_transforms.append(transforms.Normalize(mean, std))
            val_transforms.append(transforms.Normalize(mean, std))  

        train_transforms = transforms.Compose(train_transforms)
        val_transforms = transforms.Compose(val_transforms)

        _ensure_celeba_downloaded(data_dir)

        train_dataset = MyCelebA(
            data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        test_dataset = MyCelebA(
            data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
    elif dataset_name.lower() == "celeba-128":
        input_size = 128
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        train_transforms = [
            transforms.CenterCrop(178),
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ]
        
        val_transforms = [
            transforms.CenterCrop(178),
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ]

        if normalize:
            train_transforms.append(transforms.Normalize(mean, std))
            val_transforms.append(transforms.Normalize(mean, std))  

        train_transforms = transforms.Compose(train_transforms)
        val_transforms = transforms.Compose(val_transforms)

        _ensure_celeba_downloaded(data_dir)

        train_dataset = MyCelebA(
            data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        test_dataset = MyCelebA(
            data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )

    elif dataset_name.lower() == "celeba-hq":
        input_size = 256
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        train_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ]
        
        test_transforms = [
            transforms.ToTensor()
        ]

        if normalize:
            train_transforms.append(transforms.Normalize(mean, std))
            test_transforms.append(transforms.Normalize(mean, std))  

        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)

        # This loads CelebA-HQ 256x256 directly from Hugging Face (30,000 images)
        train_dataset = load_dataset("korexyz/celeba-hq-256x256", split="train")
        test_dataset = load_dataset("korexyz/celeba-hq-256x256", split="validation")

        train_dataset = HFImageDataset(train_dataset, transform=train_transforms)
        test_dataset  = HFImageDataset(test_dataset,  transform=test_transforms)
    elif dataset_name.lower() in ("oxford-flower-102"):
        # Oxford Flower 102: train(1,020)+validation(1,020) as training with size = 2,040, test(6,149) as test with size = 6,149
        input_size = 256
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        train_transforms = [
            transforms.RandomResizedCrop(
                size=256,
                scale=(0.7, 1.0),         # slightly looser than ImageNet's 0.08-1.0 to avoid too much loss of flower detail
                ratio=(3.0/4.0, 4.0/3.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.ToTensor(),
        ]
        test_transforms = [
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ]
        if normalize:
            train_transforms.append(transforms.Normalize(mean, std))
            test_transforms.append(transforms.Normalize(mean, std))
        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)

        ds = load_dataset("Donghyun99/Oxford-Flower-102")
        train_plus_test = concatenate_datasets([ds["train"], ds["validation"]])
        train_dataset = HFImageDataset(train_plus_test, transform=train_transforms)
        test_dataset = HFImageDataset(ds["test"], transform=test_transforms)
    elif dataset_name.lower() in ("animal-face", "afhq"):
        # AFHQ (Animal Faces HQ): only train split with size = 16,130 images; use same data for both train and test
        input_size = 256
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        train_transforms = [
            # transforms.CenterCrop(178),
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        test_transforms = [
            # transforms.CenterCrop(178),
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ]
        if normalize:
            train_transforms.append(transforms.Normalize(mean, std))
            test_transforms.append(transforms.Normalize(mean, std))
        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)

        ds = load_dataset("huggan/AFHQ", split="train")
        train_dataset = HFImageDataset(ds, transform=train_transforms)
        test_dataset = HFImageDataset(ds, transform=test_transforms)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return train_dataset, test_dataset, input_size
