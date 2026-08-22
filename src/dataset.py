from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=str(data_path),
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    val_dataset = datasets.CIFAR10(
        root=str(data_path),
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader

