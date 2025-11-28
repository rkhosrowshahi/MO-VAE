import numpy as np
import torch
from torchvision import datasets, transforms

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        return f"{self.avg:.4f}"

class MyCelebA(datasets.CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True


def get_dataset(dataset_name, data_dir='./data', normalize=False):
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
        input_size = 224
        
        transform_train = [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
        
        transform_test = [transforms.ToTensor()]

        if normalize:
            transform_train.append(transforms.Normalize(mean, std))
            transform_test.append(transforms.Normalize(mean, std)) # type: ignore

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        
        train_dataset = datasets.ImageNet(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.ImageNet(
            root=data_dir, train=False, download=True, transform=transform_test
        )
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
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return train_dataset, test_dataset, input_size
