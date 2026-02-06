import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

IMAGE_SIZE = 224


def get_data_transforms():
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def load_dataset_folder(data_dir='../dataset'):
    print("Loading dataset...")

    transform_train, transform_test = get_data_transforms()

    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'valid'), transform=transform_test)

    num_classes = len(train_dataset.classes)

    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Number of classes: {num_classes}")

    return train_dataset, test_dataset, num_classes


def make_dataloaders_folder(trainset, testset, batch_size, num_workers=min(8, os.cpu_count() // 2)):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    return trainloader, testloader



