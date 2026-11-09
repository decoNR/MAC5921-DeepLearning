"""Tests for the language model training loop."""

import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from ep2.data import NextCharDataset, SequenceDataset
from ep2.models import CharMLP, CharRNN
from ep2.train import evaluate, train_model, train_one_epoch

VOCAB_SIZE = 3
CONTEXT = 2
SEQ_LEN = 6

# "0 1 2 0 1 2 ...": the next token is fully determined by the current one,
# so both models should learn it almost perfectly.
PATTERN = torch.arange(300) % VOCAB_SIZE


def mlp_loaders(batch_size: int = 16):
    """Returns train and validation loaders of (window, next token) pairs."""
    train = DataLoader(NextCharDataset(PATTERN[:240], CONTEXT), batch_size, shuffle=True)
    val = DataLoader(NextCharDataset(PATTERN[240:], CONTEXT), batch_size)
    return train, val


def rnn_loaders(batch_size: int = 4):
    """Returns train and validation loaders of (sequence, shifted sequence) pairs."""
    train = DataLoader(SequenceDataset(PATTERN[:240], SEQ_LEN), batch_size, shuffle=True)
    val = DataLoader(SequenceDataset(PATTERN[240:], SEQ_LEN), batch_size)
    return train, val


def quiet(fn, *args, **kwargs):
    """Calls fn while discarding what it prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class TestEvaluate(unittest.TestCase):
    """Tests for the evaluation routine."""

    def setUp(self) -> None:
        """Builds an untrained RNN."""
        torch.manual_seed(0)
        self.model = CharRNN(VOCAB_SIZE, cell="gru", embed_dim=4, hidden_size=8)
        self.dataset = SequenceDataset(PATTERN, SEQ_LEN)

    def test_does_not_change_weights(self) -> None:
        """Evaluation leaves every parameter untouched."""
        before = [p.clone() for p in self.model.parameters()]
        evaluate(self.model, DataLoader(self.dataset, batch_size=8), torch.device("cpu"))
        for old, new in zip(before, self.model.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_untrained_loss_is_close_to_uniform(self) -> None:
        """A random model scores roughly ln V per character."""
        loss, _ = evaluate(
            self.model, DataLoader(self.dataset, batch_size=8), torch.device("cpu")
        )
        self.assertAlmostEqual(loss, math.log(VOCAB_SIZE), delta=0.3)

    def test_result_does_not_depend_on_batch_size(self) -> None:
        """Weighting by characters makes uneven last batches count correctly."""
        cpu = torch.device("cpu")
        whole = evaluate(self.model, DataLoader(self.dataset, batch_size=1000), cpu)
        uneven = evaluate(self.model, DataLoader(self.dataset, batch_size=7), cpu)
        self.assertAlmostEqual(whole[0], uneven[0], places=5)
        self.assertAlmostEqual(whole[1], uneven[1], places=5)

    def test_restores_training_mode(self) -> None:
        """Evaluation restores the previous mode."""
        self.model.train()
        evaluate(self.model, DataLoader(self.dataset, batch_size=8), torch.device("cpu"))
        self.assertTrue(self.model.training)


class TestTrainOneEpoch(unittest.TestCase):
    """Tests for a single training epoch."""

    def test_reduces_loss(self) -> None:
        """One epoch on a learnable pattern lowers the validation loss."""
        torch.manual_seed(0)
        model = CharMLP(VOCAB_SIZE, CONTEXT, embed_dim=4, hidden_size=16)
        train, val = mlp_loaders()
        cpu = torch.device("cpu")
        before, _ = evaluate(model, val, cpu)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        train_one_epoch(model, train, optimizer, cpu)
        after, _ = evaluate(model, val, cpu)
        self.assertLess(after, before)

    def test_gradient_clipping_bounds_the_update(self) -> None:
        """With SGD, clipping limits how far one step can move the weights."""
        cpu = torch.device("cpu")
        train, _ = rnn_loaders(batch_size=len(SequenceDataset(PATTERN[:240], SEQ_LEN)))
        lr, clip = 1.0, 1e-3
        torch.manual_seed(0)
        model = CharRNN(VOCAB_SIZE, cell="rnn", embed_dim=4, hidden_size=8)
        before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        train_one_epoch(model, train, optimizer, cpu, grad_clip=clip)
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        # One full-batch SGD step moves the weights by lr * ||clipped grad|| <= lr * clip.
        self.assertLessEqual((after - before).norm().item(), lr * clip + 1e-6)


class TestTrainModel(unittest.TestCase):
    """Tests for the full training loop."""

    def test_both_models_learn_the_pattern(self) -> None:
        """MLP and RNN reach near-perfect accuracy on a deterministic pattern."""
        cases = {
            "mlp": (CharMLP(VOCAB_SIZE, CONTEXT, embed_dim=4, hidden_size=16), mlp_loaders()),
            "rnn": (CharRNN(VOCAB_SIZE, cell="lstm", embed_dim=4, hidden_size=16), rnn_loaders()),
        }
        for name, (model, (train, val)) in cases.items():
            with self.subTest(model=name):
                history = quiet(
                    train_model, model, train, val, epochs=15, lr=1e-2,
                    device=torch.device("cpu"),
                )
                self.assertLess(history["val_loss"][-1], history["val_loss"][0])
                self.assertGreater(max(history["val_acc"]), 0.95)

    def test_history_has_one_entry_per_epoch(self) -> None:
        """Every metric is recorded once per epoch."""
        model = CharMLP(VOCAB_SIZE, CONTEXT, embed_dim=4, hidden_size=8)
        train, val = mlp_loaders()
        history = quiet(train_model, model, train, val, epochs=3, device=torch.device("cpu"))
        self.assertEqual(
            set(history), {"train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"}
        )
        for values in history.values():
            self.assertEqual(len(values), 3)

    def test_restores_best_weights(self) -> None:
        """The returned model scores the lowest validation loss seen in training."""
        model = CharRNN(VOCAB_SIZE, cell="rnn", embed_dim=4, hidden_size=8)
        train, val = rnn_loaders()
        # A large learning rate makes the validation loss oscillate between epochs.
        history = quiet(
            train_model, model, train, val, epochs=8, lr=0.3, device=torch.device("cpu")
        )
        final_loss, _ = evaluate(model, val, torch.device("cpu"))
        self.assertAlmostEqual(final_loss, min(history["val_loss"]), places=5)

    def test_early_stopping(self) -> None:
        """With lr=0 the loss never improves, so training stops after `patience` epochs."""
        model = CharMLP(VOCAB_SIZE, CONTEXT, embed_dim=4, hidden_size=8)
        train, val = mlp_loaders()
        history = quiet(
            train_model, model, train, val, epochs=10, lr=0.0, patience=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(history["val_loss"]), 3)

    def test_same_seed_is_reproducible(self) -> None:
        """Two runs with the same seed produce the same history."""
        histories = []
        for _ in range(2):
            torch.manual_seed(0)
            model = CharMLP(VOCAB_SIZE, CONTEXT, embed_dim=4, hidden_size=8)
            train = DataLoader(
                NextCharDataset(PATTERN[:240], CONTEXT), 16, shuffle=True,
                generator=torch.Generator().manual_seed(1),
            )
            val = DataLoader(NextCharDataset(PATTERN[240:], CONTEXT), 16)
            history = quiet(
                train_model, model, train, val, epochs=2, device=torch.device("cpu")
            )
            histories.append(history["val_loss"])
        self.assertEqual(histories[0], histories[1])


if __name__ == "__main__":
    unittest.main()
