import torch
from torchvision import datasets, transforms


class SVHNDataLoader:
    def __init__(self, batch_size=512, num_workers=12, one_hot=False, flatten=False):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_set = datasets.SVHN(
            root="./data", split="train", download=True, transform=transform
        )

        test_set = datasets.SVHN(
            root="./data", split="test", download=True, transform=transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, test_loader


if __name__ == "__main__":
    batch_size = 512
    num_workers = 12

    data_loader = SVHNDataLoader(batch_size=batch_size, num_workers=num_workers)
    train_loader, test_loader = data_loader.get_data_loaders()
    print(len(train_loader))
