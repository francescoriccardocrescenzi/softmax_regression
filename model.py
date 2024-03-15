"""Softmax regression model."""
import torch
from torch import nn


class SoftmaxModel(nn.Module):
    """Model that uses softmax regression to solve a linear classification problem."""
    def __init__(self, in_features: int, out_features: int):
        """Constructor for SoftmaxModel.

        :param in_features: number of input features
        :param out_features: number of output features
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        """Forward pass through the network."""
        x = self.flatten(x)
        x = self.linear(x)
        return x
