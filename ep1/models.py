"""Models compared in this assignment and helpers to size them."""

import torch
from torch import nn

# MNIST images are 1x28x28, which a fully connected net sees as 784 features.
IMAGE_SIZE = 28
INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10


def count_parameters(model: nn.Module) -> int:
    """Counts the trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    """A fully connected network."""

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


class CNN(nn.Module):
    """A convolutional network."""

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Builds a CNN for square image_size x image_size images."""
        super().__init__()
        # Each of the two poolings halves both sides of the feature map.
        pooled_size = image_size // 4
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * pooled_size * pooled_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps a batch of images to a batch of class logits."""
        return self.layers(x)
