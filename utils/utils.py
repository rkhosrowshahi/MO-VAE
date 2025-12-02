import numpy as np
import torch
from torchvision import datasets, transforms
from datasets import load_dataset
from torch.utils.data import Dataset

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
    
    Supports CIFAR-10, CIFAR-100, ImageNet, and CelebA datasets. Applies appropriate
    transforms including data augmentation for training sets and optional normalization.
    
    Args:
        dataset_name: Name of the dataset. Supported values:
                     - 'CIFAR10': CIFAR-10 dataset (32x32 images, 10 classes)
                     - 'CIFAR100': CIFAR-100 dataset (32x32 images, 100 classes)
                     - 'ImageNet': ImageNet dataset (224x224 images, 1000 classes)
                     - 'CelebA': CelebA face dataset (64x64 images)
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
        
        # train_transforms = [
        #     transforms.RandomResizedCrop(256),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ]
        
        # test_transforms = [
        #     transforms.Resize(256),
        #     transforms.CenterCrop(256),
        #     transforms.ToTensor(),
        # ]

        # if normalize:
        #     train_transforms.append(transforms.Normalize(mean, std))
        #     test_transforms.append(transforms.Normalize(mean, std))

        # train_transforms = transforms.Compose(train_transforms)
        # test_transforms = transforms.Compose(test_transforms)
        
        # train_dataset = datasets.ImageNet(
        #     root=data_dir, split='train', transform=transform_train
        # )
        # test_dataset = datasets.ImageNet(
        #     root=data_dir, split='val', transform=transform_test
        # )

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

        # def train_transform_example(examples):
        #     images = [train_transforms(img.convert("RGB")) for img in examples["image"]]
        #     return {"image": images, "labels": examples["label"]}

        # def test_transform_example(examples):
        #     images = [test_transforms(img.convert("RGB")) for img in examples["image"]]
        #     return {"image": images, "labels": examples["label"]}

        train_dataset = HFImageDataset(train_dataset, transform=train_transforms)
        test_dataset  = HFImageDataset(test_dataset,  transform=test_transforms)
    elif dataset_name.lower() == "celeba":
        input_size = 64
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        train_transforms = [transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(input_size),
                                              transforms.ToTensor(),]
        
        val_transforms = [
                                            transforms.CenterCrop(148),
                                            transforms.Resize(input_size),
                                            transforms.ToTensor(),]

        if normalize:
            train_transforms.append(transforms.Normalize(mean, std))
            val_transforms.append(transforms.Normalize(mean, std))  

        train_transforms = transforms.Compose(train_transforms)
        val_transforms = transforms.Compose(val_transforms)

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
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(178),
            transforms.Resize(input_size),
            transforms.ToTensor()
        ]
        
        val_transforms = [
            transforms.CenterCrop(178),
            transforms.Resize(input_size),
            transforms.ToTensor()
        ]

        if normalize:
            train_transforms.append(transforms.Normalize(mean, std))
            val_transforms.append(transforms.Normalize(mean, std))  

        train_transforms = transforms.Compose(train_transforms)
        val_transforms = transforms.Compose(val_transforms)

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
            # transforms.RandomHorizontalFlip(),
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
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return train_dataset, test_dataset, input_size
