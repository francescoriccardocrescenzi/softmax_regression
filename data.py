from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from typing import Tuple


def get_MNIST_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Access the MNIST dataset and initialize a training dataloader and a testing dataloader with it.

    The MNIST dataset is downloaded if not present.

    :param batch_size: batch size for the dataloaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    testing_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)

    return training_dataloader, testing_dataloader
