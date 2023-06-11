import torch
from torchvision import datasets, transforms


class CIFAR10DataLoader:
    def __init__(self, batch_size=512, num_workers=12):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self):
        # Load the CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(
            root="../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )

        test_dataset = datasets.CIFAR10(
            root="../data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )

        # Split the training dataset into training and validation datasets
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(0.8 * num_train)  # 80% for training, 20% for validation
        train_indices, val_indices = indices[:split], indices[split:]

        # Create data loaders for training, validation, and testing datasets
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
            num_workers=self.num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices),
            num_workers=self.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    batch_size = 64
    num_workers = 2

    data_loader = CIFAR10DataLoader(batch_size, num_workers)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    print(train_loader)
