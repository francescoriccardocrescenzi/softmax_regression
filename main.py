from data import get_MNIST_dataloaders
from visualization import print_image_from_MNIST_dataloader

# Hyperparameters
batch_size = 100
training_dataloader, testing_dataloader = get_MNIST_dataloaders(batch_size)

print_image_from_MNIST_dataloader(training_dataloader)


