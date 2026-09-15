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


# Comparing the two architectures is only fair if neither gets more capacity
# than the other, so the MLP is sized to match the CNN's budget of 52,138
# parameters, computed layer by layer as follows:
#
#   Conv2d(1, 8, 3):             (1 * 3 * 3 + 1) * 8      =     80
#   Conv2d(8, 16, 3):            (8 * 3 * 3 + 1) * 16     =  1,168
#   Linear(16 * 7 * 7, 64):      (784 + 1) * 64           = 50,240
#   Linear(64, 10):              (64 + 1) * 10            =    650
#   --------------------------------------------------------------
#   Total CNN:                                              52,138
#
# An MLP on MNIST with hidden size h holds:
#   Linear(784, h):              (784 + 1) * h            =   785h
#   Linear(h, 10):               (h + 1) * 10             = 10h + 10
#   --------------------------------------------------------------
#   Total MLP:                                            795h + 10
#
# Setting 795h + 10 = 52,138 gives h = 65.57, so the nearest whole layer size
# is h = 66, yielding 795 * 66 + 10 = 52,480 parameters (+0.66% over the CNN).
MNIST_HIDDEN_SIZE = 66

# Both models stay within this fraction of each other's parameter count.
PARAMETER_TOLERANCE = 0.02


def make_mlp() -> MLP:
    """Builds the fully connected network compared in the experiment."""
    return MLP(hidden_size=MNIST_HIDDEN_SIZE)


def make_cnn() -> CNN:
    """Builds the convolutional network compared in the experiment."""
    return CNN(image_size=IMAGE_SIZE)
