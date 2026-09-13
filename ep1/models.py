"""Models compared in this assignment and helpers to size them."""

import torch
from torch import nn

# MNIST images are 1x28x28, which a fully connected net sees as 784 features.
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10


def count_parameters(model: nn.Module) -> int:
    """Counts the trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    """A fully connected network with a single hidden layer."""

    def __init__(
        self,
        hidden_size: int,
        input_size: int = INPUT_SIZE,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Builds the input_size -> hidden_size -> num_classes stack."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps a batch of images to a batch of class logits."""
        return self.layers(x)
