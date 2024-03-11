import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.utils.data import DataLoader


def print_image_from_MNIST_dataloader(dataloader: DataLoader):
    """Print one image from the given MNIST dataset.

    Acknowledgement: This function uses code generated by chatgpt 3.5.
    """

    # Get the images from the dataloader, mount them into a grid, un-normalize them,
    # convert them to numpy arrays, transpose the arrays for displaying, and, finally, display them.
    images, labels = next(iter(dataloader))
    grid = torchvision.utils.make_grid(images)
    grid = grid / 2 + 0.5
    numpy_grid = grid.numpy()
    plt.imshow(np.transpose(numpy_grid, (1, 2, 0)))
    plt.show()