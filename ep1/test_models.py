"""Tests for the models compared in this assignment."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from ep1.models import INPUT_SIZE, NUM_CLASSES, MLP, count_parameters

BATCH_SIZE = 3
TOY_HIDDEN_SIZE = 5


def make_image_batch(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Builds a batch of MNIST-shaped images filled with random pixels."""
    return torch.rand(batch_size, 1, 28, 28)


class TestCountParameters(unittest.TestCase):
    """Tests for count_parameters."""

    def test_counts_weights_and_biases(self) -> None:
        """A Linear(4, 3) holds 4*3 weights plus 3 biases."""
        self.assertEqual(count_parameters(nn.Linear(4, 3)), 15)

    def test_ignores_frozen_parameters(self) -> None:
        """Parameters excluded from training are excluded from the count."""
        layer = nn.Linear(4, 3)
        layer.bias.requires_grad_(False)

        self.assertEqual(count_parameters(layer), 12)

    def test_sums_over_submodules(self) -> None:
        """A stack of layers counts as the sum of its parts."""
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

        self.assertEqual(count_parameters(model), 15 + 8)


class TestMLP(unittest.TestCase):
    """Tests for MLP."""

    def setUp(self) -> None:
        """Creates the small MLP shared by these tests."""
        self.model = MLP(hidden_size=TOY_HIDDEN_SIZE)

    def test_forward_gives_one_logit_per_class(self) -> None:
        """A batch of images becomes a batch of class logits."""
        logits = self.model(make_image_batch())

        self.assertEqual(logits.shape, (BATCH_SIZE, NUM_CLASSES))

    def test_forward_accepts_a_single_sample(self) -> None:
        """The batch dimension may hold a single image."""
        logits = self.model(make_image_batch(1))

        self.assertEqual(logits.shape, (1, NUM_CLASSES))

    def test_logits_are_not_probabilities(self) -> None:
        """The output is unnormalized, as CrossEntropyLoss expects."""
        logits = self.model(make_image_batch())

        sums = logits.sum(dim=1)

        self.assertFalse(torch.allclose(sums, torch.ones_like(sums)))

    def test_parameter_count_matches_the_layer_sizes(self) -> None:
        """The MLP holds exactly the two Linear layers it declares."""
        expected = (INPUT_SIZE + 1) * TOY_HIDDEN_SIZE + (
            TOY_HIDDEN_SIZE + 1
        ) * NUM_CLASSES

        self.assertEqual(count_parameters(self.model), expected)


if __name__ == "__main__":
    unittest.main()
