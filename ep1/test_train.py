"""Tests for the training and evaluation routines."""

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ep1.train import evaluate, set_seed, summarize, train_model, train_one_epoch


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


class TestTrainModel(unittest.TestCase):
    """Tests for train_model."""

    def setUp(self) -> None:
        """Sets up a toy dataset, dataloader, model, and device."""
        self.device = torch.device("cpu")
        self.dataset = make_toy_dataset()
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        self.model = nn.Linear(2, 2)

    def test_history_keys_and_lengths(self) -> None:
        """History contains all 5 keys with length equal to epochs when val_loader is given."""
        epochs = 3
        history = train_model(
            self.model,
            self.loader,
            val_loader=self.loader,
            epochs=epochs,
            lr=0.01,
            device=self.device,
            seed=42,
        )

        expected_keys = {"train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"}
        self.assertEqual(set(history.keys()), expected_keys)
        for key in expected_keys:
            self.assertEqual(len(history[key]), epochs)

    def test_same_seed_produces_identical_history(self) -> None:
        """Training two identical models with the same seed yields the same history."""
        torch.manual_seed(99)
        model_a = nn.Linear(2, 2)
        model_b = nn.Linear(2, 2)
        model_b.load_state_dict(model_a.state_dict())

        history_a = train_model(
            model_a,
            self.loader,
            val_loader=self.loader,
            epochs=3,
            lr=0.01,
            device=self.device,
            seed=123,
        )
        history_b = train_model(
            model_b,
            self.loader,
            val_loader=self.loader,
            epochs=3,
            lr=0.01,
            device=self.device,
            seed=123,
        )

        self.assertEqual(history_a["train_loss"], history_b["train_loss"])
        self.assertEqual(history_a["train_acc"], history_b["train_acc"])
        self.assertEqual(history_a["val_loss"], history_b["val_loss"])
        self.assertEqual(history_a["val_acc"], history_b["val_acc"])

    def test_works_without_val_loader(self) -> None:
        """Without val_loader, validation keys remain empty while training keys have length epochs."""
        epochs = 3
        history = train_model(
            self.model,
            self.loader,
            val_loader=None,
            epochs=epochs,
            lr=0.01,
            device=self.device,
            seed=42,
        )

        self.assertEqual(len(history["train_loss"]), epochs)
        self.assertEqual(len(history["train_acc"]), epochs)
        self.assertEqual(len(history["epoch_time"]), epochs)
        self.assertEqual(history["val_loss"], [])
        self.assertEqual(history["val_acc"], [])


class TestSummarize(unittest.TestCase):
    """Tests for summarize."""

    def test_metrics_calculation(self) -> None:
        """Synthetic hand-crafted history produces exact expected summary metrics."""
        history = {
            "train_loss": [0.6, 0.4, 0.2, 0.1],
            "train_acc": [0.7, 0.85, 0.95, 0.98],
            "val_loss": [0.5, 0.35, 0.25, 0.2],
            "val_acc": [0.70, 0.90, 0.995, 1.0],
            "epoch_time": [2.0, 4.0, 3.0, 1.0],
        }
        summary = summarize(
            history,
            model_name="MLP",
            num_params=52480,
            test_acc=0.975,
        )

        self.assertEqual(summary["model_name"], "MLP")
        self.assertEqual(summary["num_params"], 52480)
        self.assertEqual(summary["test_acc"], 0.975)
        self.assertAlmostEqual(summary["best_val_acc"], 1.0)
        self.assertEqual(summary["best_val_epoch"], 4)
        self.assertEqual(summary["epochs_to_convergence"], 3)
        self.assertAlmostEqual(summary["total_time"], 10.0)
        self.assertAlmostEqual(summary["mean_epoch_time"], 2.5)

    def test_json_serializability(self) -> None:
        """The returned summary dictionary is JSON serializable."""
        history = {
            "train_loss": [0.5, 0.3],
            "train_acc": [0.8, 0.9],
            "val_loss": [0.5, 0.3],
            "val_acc": [0.85, 0.95],
            "epoch_time": [1.5, 2.5],
        }
        summary = summarize(history, "CNN", 52138, 0.98)
        encoded = json.dumps(summary)
        decoded = json.loads(encoded)
        self.assertEqual(decoded, summary)

    def test_immediate_convergence(self) -> None:
        """When the first epoch achieves >= 99% of best accuracy, epochs_to_convergence is 1."""
        history = {
            "train_loss": [0.1, 0.05],
            "train_acc": [0.98, 0.99],
            "val_loss": [0.1, 0.05],
            "val_acc": [0.98, 0.985],
            "epoch_time": [2.0, 2.0],
        }
        summary = summarize(history, "CNN", 52138, 0.98)
        self.assertEqual(summary["best_val_epoch"], 2)
        self.assertEqual(summary["epochs_to_convergence"], 1)

    def test_empty_validation_and_timing(self) -> None:
        """Handles empty val_acc and epoch_time safely."""
        history = {
            "train_loss": [0.5],
            "train_acc": [0.8],
            "val_loss": [],
            "val_acc": [],
            "epoch_time": [],
        }
        summary = summarize(history, "MLP", 100, 0.5)
        self.assertEqual(summary["best_val_acc"], 0.0)
        self.assertEqual(summary["best_val_epoch"], 0)
        self.assertEqual(summary["epochs_to_convergence"], 0)
        self.assertEqual(summary["total_time"], 0.0)
        self.assertEqual(summary["mean_epoch_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
