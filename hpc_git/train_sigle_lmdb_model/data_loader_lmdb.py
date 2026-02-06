import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import io
import lmdb
import pickle

IMAGE_SIZE = 224


def get_data_transforms():
    transform_train = transforms.Compose([
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


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None  # Do not open here!
        self.keys = None

        temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with temp_env.begin() as txn:
            self.keys = [key for key, _ in txn.cursor()]
        temp_env.close()

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, 
                             readahead=False, meminit=False)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        key = self.keys[idx]
        with self.env.begin() as txn:
            data = txn.get(key)
            if data is None:
                raise IndexError(f"Key {key} not found")
            img_bytes, label = pickle.loads(data)
            
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, int(label)


def load_dataset_lmdb(data_dir='../dataset', num_classes=365):
    print("Loading LMDB dataset...")

    transform_train, transform_test = get_data_transforms()

    train_lmdb_path = os.path.join(data_dir, 'train_lmdb')
    val_lmdb_path   = os.path.join(data_dir, 'val_lmdb')

    train_dataset = LMDBDataset(train_lmdb_path, transform=transform_train)
    test_dataset  = LMDBDataset(val_lmdb_path, transform=transform_test)

    print(f"Dataset loaded successfully:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    print(f"  Fixed classes: {num_classes}")

    return train_dataset, test_dataset, num_classes



def make_dataloaders_lmdb(trainset, testset, batch_size, num_workers=min(8, os.cpu_count() // 2)):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    return trainloader, testloader



