from torch import nn
from data import MNIST
from model import SoftmaxModel
from torch.utils.data import DataLoader
from torch.optim import SGD
from train import Trainer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# Hyperparameters
batch_size = 100
learning_rate = 0.1
weight_decay = 0.001
num_epochs = 10

# Data
mnist = MNIST()
training_dataloader = DataLoader(mnist.training_dataset, batch_size=batch_size, shuffle=True)
testing_dataloader = DataLoader(mnist.testing_dataset, batch_size=batch_size, shuffle=True)

# Model
model = SoftmaxModel(in_features=mnist.features, out_features=mnist.classes)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer = SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Trainer
trainer = Trainer(model, training_dataloader, testing_dataloader, loss_function, optimizer)

# Create summary writer to log losses to TensorBoard.
# Use a timestamp to differentiate between runs in the TensorBoard logs.
timestamp = int(datetime.now().timestamp())
with SummaryWriter(log_dir=f"logs/{timestamp}") as summary_writer:

    # Execute training loop.
    trainer.train(epochs=num_epochs, summary_writer=summary_writer)

