import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    return transform_train, transform_test


def load_cifar100_dataset(data_dir='./cifar100'):
    print("Loading CIFAR-100 dataset...")

    transform_train, transform_test = get_data_transforms()

    if os.path.exists(data_dir):
        print("Using existing CIFAR-100 data.")
        print(f"ATTANTION: DATA FOUND IN {data_dir}")
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                 download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                                download=False, transform=transform_test)
    else:
        raise RuntimeError(f"no data")

    print(f"CIFAR-100 loaded: Train {len(trainset)}, Test {len(testset)}")

    return trainset, testset


def make_dataloaders(trainset, testset, batch_size, num_workers=min(8, os.cpu_count() // 2)):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    return trainloader, testloader

