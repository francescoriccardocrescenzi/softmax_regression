"""Trainer object to train linear regression models."""
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    """Train an instance of linear regression model.

    Every instance is initialized with a given training dataloader, testing dataloader, loss function and optimizer.
    Use these objects to execute an appropriate training loop.

    Trainer can also log training and testing losses to tensor board.
    """

    def __init__(self, model: nn.Module, training_dataloader: DataLoader, testing_dataloader: DataLoader,
                 loss_function, optimizer):
        """Constructor for Trainer.

        :param model: model to be trained.
        :param training_dataloader: dataloader for the training dataset.
        :param testing_dataloader: dataloader for the testing dataset.
        :param loss_function: loss function.
        :param optimizer: optimizer.
        """
        self.model = model
        self.training_dataloader = training_dataloader
        self.testing_dataloader = testing_dataloader
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train_one_epoch(self, current_epoch: int, summary_writer: SummaryWriter = None):
        """Train the model for a single epoch.

        If self.summary_writer is defined, log training and testing losses to it.

        :param current_epoch: current epoch, needed to log losses to correct index.
        :param summary_writer: if not None, the writer will be used to log the training loss.
        """
        for batch_index, data in enumerate(self.training_dataloader):
            instances, labels = data
            self.model.train()
            predictions = self.model(instances)
            loss = self.loss_function(predictions, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if summary_writer is not None:
                summary_writer.add_scalar('training_loss', loss.item(),
                                          current_epoch * len(self.training_dataloader) + batch_index)

    def test(self, current_epoch, summary_writer: SummaryWriter = None):
        """If self.summary_writer is defined, test the model on the testing data and log the associated losses.

        :param current_epoch: current epoch, needed to log losses to correct index.
        :param summary_writer: if not None, the writer will be used to log the training loss.
        """
        if summary_writer is not None:
            with torch.inference_mode():
                for batch_index, data in enumerate(self.testing_dataloader):
                    instances, labels = data
                    predictions = self.model(instances)
                    loss = self.loss_function(predictions, labels)

                summary_writer.add_scalar('testing_losses', loss.item(),
                                          current_epoch * len(self.training_dataloader) + batch_index)

    def train(self, epochs: int, summary_writer: SummaryWriter = None):
        """Execute the training loop, test the model, and log both testing and training losses.

        :param epochs: number of epochs.
        :param summary_writer: if not None, the writer will be used to log the training loss.
        """
        for epoch in range(epochs):
            print(f"epoch = {epoch}")
            self.train_one_epoch(current_epoch=epoch, summary_writer=summary_writer)
            self.test(current_epoch=epoch, summary_writer=summary_writer)
