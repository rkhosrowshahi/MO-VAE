import numpy as np
import torch
from torchvision import datasets, transforms

from models import VAE, VQVAE

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def get_dataset(dataset_name, normalize=False):
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
            transform_train.append(transforms.Normalize(mean, std))
            transform_test.append(transforms.Normalize(mean, std))

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
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
            transform_train.append(transforms.Normalize(mean, std))
            transform_test.append(transforms.Normalize(mean, std))

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        
        train_dataset = datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 100
        class_names = [f'class {i}' for i in range(100)]
    elif dataset_name.lower() == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        input_size = 224
        
        transform_train = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        
        transform_test = [transforms.ToTensor()]

        if normalize:
            transform_train.append(transforms.Normalize(mean, std))
            transform_test.append(transforms.Normalize(mean, std)) # type: ignore

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        
        train_dataset = datasets.ImageNet(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.ImageNet(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 1000
        class_names = [f'class {i}' for i in range(1000)]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return train_dataset, test_dataset, input_size, num_classes, class_names
