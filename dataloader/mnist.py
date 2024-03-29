import torch
from torchvision import datasets, transforms


class MNISTDataLoader:
    def __init__(
        self,
        batch_size=512,
        num_workers=12,
        validation=True,
        one_hot=False,
        flatten=False,
        root="data",
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.one_hot = one_hot
        self.flatten = flatten
        self.root = root
        self.validation = validation

    def one_hot_encoder(self, batch):
        images, labels = zip(*batch)
        labels = torch.tensor(labels)
        return torch.stack(images), torch.nn.functional.one_hot(labels, 10).float()

    def get_data_loaders(self):
        collate_fn = self.one_hot_encoder if self.one_hot else None

        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1) if self.flatten else x),
            ]
        )
        # Load the MNIST dataset
        train_dataset = datasets.MNIST(
            root=self.root, train=True, download=True, transform=transform
        )

        test_dataset = datasets.MNIST(
            root=self.root, train=False, download=True, transform=transform
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        if self.validation:
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
                collate_fn=collate_fn,
            )
            val_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices),
                num_workers=self.num_workers,
                collate_fn=collate_fn,
            )
            return train_loader, val_loader, test_loader
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
            )
            return train_loader, test_loader


if __name__ == "__main__":
    batch_size = 64
    num_workers = 2

    data_loader = MNISTDataLoader(batch_size, num_workers)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    print(train_loader)
