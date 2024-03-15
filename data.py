"""Module used to access and visualize data from the MNIST dataset."""
import torchvision


class MNIST:
    def __init__(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        self.training_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        self.testing_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        self.features = 28*28
        self.classes = 10

