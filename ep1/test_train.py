"""Tests for the training and evaluation routines."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ep1.train import evaluate, set_seed, train_one_epoch


def make_toy_dataset() -> TensorDataset:
    """Builds a small, linearly separable 2D classification dataset."""
    features = torch.tensor(
        [
            [3.0, 1.0],
            [2.0, 0.5],
            [4.0, 2.0],
            [-3.0, -1.0],
            [-2.0, -0.5],
            [-4.0, -2.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    return TensorDataset(features, targets)




class TestTrainOneEpoch(unittest.TestCase):
    """Tests for train_one_epoch."""

    def setUp(self) -> None:
        """Sets up a toy dataset, dataloader, model, criterion, and optimizer."""
        set_seed(42)
        self.device = torch.device("cpu")
        self.dataset = make_toy_dataset()
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        self.model = nn.Linear(2, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_leaves_model_in_training_mode(self) -> None:
        """train_one_epoch sets model.training to True."""
        self.model.eval()
        train_one_epoch(
            self.model, self.loader, self.criterion, self.optimizer, self.device
        )

        self.assertTrue(self.model.training)

    def test_returns_valid_metrics(self) -> None:
        """Loss is non-negative and accuracy is between 0.0 and 1.0."""
        loss, acc = train_one_epoch(
            self.model, self.loader, self.criterion, self.optimizer, self.device
        )

        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


class TestEvaluate(unittest.TestCase):
    """Tests for evaluate."""

    def setUp(self) -> None:
        """Sets up a toy dataset, dataloader, model, and criterion."""
        self.device = torch.device("cpu")
        self.dataset = make_toy_dataset()
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        self.model = nn.Linear(2, 2)
        self.criterion = nn.CrossEntropyLoss()

    def test_does_not_alter_weights(self) -> None:
        """evaluate does not change model parameters."""
        initial_params = [p.clone() for p in self.model.parameters()]

        evaluate(self.model, self.loader, self.criterion, self.device)

        for p_before, p_after in zip(initial_params, self.model.parameters()):
            self.assertTrue(torch.equal(p_before, p_after))

    def test_leaves_model_in_eval_mode(self) -> None:
        """evaluate sets model.training to False."""
        self.model.train()
        evaluate(self.model, self.loader, self.criterion, self.device)

        self.assertFalse(self.model.training)

    def test_accuracy_is_one_on_trivially_separable_data(self) -> None:
        """A model with separating hyperplane achieves 100% accuracy."""
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[2.0, 0.0], [-2.0, 0.0]]))
            self.model.bias.zero_()

        _, acc = evaluate(self.model, self.loader, self.criterion, self.device)

        self.assertEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
