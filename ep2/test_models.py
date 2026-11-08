"""Tests for the character-level language models."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from ep1.models import count_parameters
from ep2.models import CELL_TYPES, GATES_PER_CELL, CharMLP, CharRNN, sample_next

VOCAB_SIZE = 7
BATCH_SIZE = 3
CONTEXT = 4
SEQ_LEN = 5
EMBED_DIM = 6
HIDDEN_SIZE = 8


def overfit_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple:
    """Trains on a single batch and returns the (initial, final) losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses[0], losses[-1]


class TestSampleNext(unittest.TestCase):
    """Tests for temperature sampling."""

    def test_returns_one_id_per_row(self) -> None:
        """(B, V) logits yield B token ids."""
        ids = sample_next(torch.zeros(BATCH_SIZE, VOCAB_SIZE))
        self.assertEqual(ids.shape, (BATCH_SIZE,))
        self.assertTrue(((ids >= 0) & (ids < VOCAB_SIZE)).all())

    def test_low_temperature_is_almost_greedy(self) -> None:
        """A tiny temperature always picks the most likely token."""
        logits = torch.tensor([[0.0, 1.0, 0.5]]).repeat(50, 1)
        ids = sample_next(logits, temperature=1e-3)
        self.assertTrue((ids == 1).all())

    def test_same_generator_seed_is_reproducible(self) -> None:
        """Sampling with equally seeded generators gives the same ids."""
        logits = torch.zeros(20, VOCAB_SIZE)
        first = sample_next(logits, generator=torch.Generator().manual_seed(0))
        second = sample_next(logits, generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(first, second))

    def test_non_positive_temperature_raises(self) -> None:
        """Temperature must be positive."""
        with self.assertRaises(ValueError):
            sample_next(torch.zeros(1, VOCAB_SIZE), temperature=0.0)


class TestCharMLP(unittest.TestCase):
    """Tests for the fixed-window MLP."""

    def setUp(self) -> None:
        """Builds a small MLP with a fixed seed."""
        torch.manual_seed(0)
        self.model = CharMLP(
            VOCAB_SIZE, CONTEXT, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE
        )

    def test_output_shape(self) -> None:
        """(B, context) windows map to (B, V) logits."""
        x = torch.randint(VOCAB_SIZE, (BATCH_SIZE, CONTEXT))
        self.assertEqual(self.model(x).shape, (BATCH_SIZE, VOCAB_SIZE))

    def test_parameter_count(self) -> None:
        """Embedding V*E, hidden layer (k*E + 1)*H and output layer (H + 1)*V."""
        expected = (
            VOCAB_SIZE * EMBED_DIM
            + (CONTEXT * EMBED_DIM + 1) * HIDDEN_SIZE
            + (HIDDEN_SIZE + 1) * VOCAB_SIZE
        )
        self.assertEqual(count_parameters(self.model), expected)

    def test_parameters_grow_linearly_with_context(self) -> None:
        """Each extra character of context adds E*H weights to the first layer."""
        wider = CharMLP(
            VOCAB_SIZE, CONTEXT + 1, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE
        )
        difference = count_parameters(wider) - count_parameters(self.model)
        self.assertEqual(difference, EMBED_DIM * HIDDEN_SIZE)

    def test_position_matters(self) -> None:
        """Swapping two characters in the window changes the prediction."""
        x = torch.tensor([[0, 1, 2, 3]])
        swapped = torch.tensor([[1, 0, 2, 3]])
        self.assertFalse(torch.allclose(self.model(x), self.model(swapped)))

    def test_wrong_window_shape_raises(self) -> None:
        """Windows must have shape (B, context)."""
        for shape in ((BATCH_SIZE, CONTEXT + 1), (BATCH_SIZE, CONTEXT - 1), (CONTEXT,)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    self.model(torch.zeros(shape, dtype=torch.long))

    def test_can_overfit_a_batch(self) -> None:
        """Training on a single batch drives its loss down."""
        x = torch.randint(VOCAB_SIZE, (BATCH_SIZE, CONTEXT))
        y = torch.randint(VOCAB_SIZE, (BATCH_SIZE,))
        initial, final = overfit_loss(self.model, x, y)
        self.assertLess(final, initial * 0.2)


class TestCharMLPGenerate(unittest.TestCase):
    """Tests for text generation with the MLP."""

    def setUp(self) -> None:
        """Builds a small MLP with a fixed seed."""
        torch.manual_seed(0)
        self.model = CharMLP(
            VOCAB_SIZE, CONTEXT, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE
        )
        self.prompt = torch.tensor([0, 3, 1, 2, 5])

    def test_appends_tokens_after_prompt(self) -> None:
        """The output keeps the prompt and adds max_new_tokens valid ids."""
        out = self.model.generate(self.prompt, max_new_tokens=10)
        self.assertEqual(out.shape, (len(self.prompt) + 10,))
        self.assertTrue(torch.equal(out[: len(self.prompt)], self.prompt))
        self.assertTrue(((out >= 0) & (out < VOCAB_SIZE)).all())

    def test_short_prompt_raises(self) -> None:
        """The prompt must fill at least one window."""
        with self.assertRaises(ValueError):
            self.model.generate(self.prompt[: CONTEXT - 1], max_new_tokens=1)

    def test_restores_training_mode(self) -> None:
        """Generation runs in eval mode and restores the previous mode."""
        self.model.train()
        self.model.generate(self.prompt, max_new_tokens=2)
        self.assertTrue(self.model.training)

    def test_reproducible_with_generator(self) -> None:
        """Equally seeded generators yield the same text."""
        outs = [
            self.model.generate(
                self.prompt, 20, generator=torch.Generator().manual_seed(3)
            )
            for _ in range(2)
        ]
        self.assertTrue(torch.equal(outs[0], outs[1]))


class TestCharRNN(unittest.TestCase):
    """Tests for the recurrent models, run for every cell type."""

    def build(self, cell: str, num_layers: int = 1) -> CharRNN:
        """Builds a small recurrent model with a fixed seed."""
        torch.manual_seed(0)
        return CharRNN(
            VOCAB_SIZE,
            cell=cell,
            embed_dim=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=num_layers,
        )

    def test_output_shape(self) -> None:
        """(B, T) tokens map to (B, T, V) logits."""
        x = torch.randint(VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        for cell in CELL_TYPES:
            with self.subTest(cell=cell):
                logits = self.build(cell)(x)
                self.assertEqual(logits.shape, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))

    def test_parameter_count(self) -> None:
        """Each layer has gates * (in*H + H*H + 2H) weights (PyTorch keeps two biases)."""
        for cell in CELL_TYPES:
            for num_layers in (1, 2):
                with self.subTest(cell=cell, num_layers=num_layers):
                    gates = GATES_PER_CELL[cell]
                    recurrent = 0
                    input_size = EMBED_DIM
                    for _ in range(num_layers):
                        recurrent += gates * (
                            input_size * HIDDEN_SIZE
                            + HIDDEN_SIZE * HIDDEN_SIZE
                            + 2 * HIDDEN_SIZE
                        )
                        input_size = HIDDEN_SIZE
                    expected = (
                        VOCAB_SIZE * EMBED_DIM
                        + recurrent
                        + (HIDDEN_SIZE + 1) * VOCAB_SIZE
                    )
                    model = self.build(cell, num_layers)
                    self.assertEqual(count_parameters(model), expected)

    def test_parameters_do_not_depend_on_sequence_length(self) -> None:
        """Weights are shared across time, so any sequence length works."""
        model = self.build("gru")
        for length in (1, SEQ_LEN, 3 * SEQ_LEN):
            with self.subTest(length=length):
                x = torch.randint(VOCAB_SIZE, (BATCH_SIZE, length))
                self.assertEqual(model(x).shape[1], length)

    def test_is_causal(self) -> None:
        """Changing a future token does not change earlier predictions."""
        for cell in CELL_TYPES:
            with self.subTest(cell=cell):
                model = self.build(cell).eval()
                x = torch.randint(VOCAB_SIZE, (1, SEQ_LEN))
                changed = x.clone()
                changed[0, -1] = (changed[0, -1] + 1) % VOCAB_SIZE
                self.assertTrue(
                    torch.allclose(model(x)[:, :-1], model(changed)[:, :-1])
                )

    def test_early_tokens_influence_late_predictions(self) -> None:
        """Unlike a fixed window, the first token still affects the last prediction."""
        model = self.build("lstm").eval()
        x = torch.randint(VOCAB_SIZE, (1, 3 * SEQ_LEN))
        changed = x.clone()
        changed[0, 0] = (changed[0, 0] + 1) % VOCAB_SIZE
        self.assertFalse(torch.allclose(model(x)[:, -1], model(changed)[:, -1]))

    def test_state_carries_context(self) -> None:
        """Feeding a sequence in two halves with the carried state equals one pass."""
        for cell in CELL_TYPES:
            with self.subTest(cell=cell):
                model = self.build(cell).eval()
                x = torch.randint(VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
                full, _ = model.forward_with_state(x)
                first, hidden = model.forward_with_state(x[:, :2])
                second, _ = model.forward_with_state(x[:, 2:], hidden)
                self.assertTrue(
                    torch.allclose(full, torch.cat([first, second], dim=1), atol=1e-6)
                )

    def test_wrong_input_shape_raises(self) -> None:
        """Inputs must have shape (B, T)."""
        with self.assertRaises(ValueError):
            self.build("rnn")(torch.zeros(SEQ_LEN, dtype=torch.long))

    def test_unknown_cell_raises(self) -> None:
        """Only rnn, gru and lstm are supported."""
        with self.assertRaises(ValueError):
            CharRNN(VOCAB_SIZE, cell="transformer")

    def test_can_overfit_a_batch(self) -> None:
        """Training on a single batch drives its loss down."""
        x = torch.randint(VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        for cell in CELL_TYPES:
            with self.subTest(cell=cell):
                initial, final = overfit_loss(self.build(cell), x, y)
                self.assertLess(final, initial * 0.2)


class TestCharRNNGenerate(unittest.TestCase):
    """Tests for text generation with the recurrent models."""

    def build(self, cell: str) -> CharRNN:
        """Builds a small recurrent model with a fixed seed."""
        torch.manual_seed(0)
        return CharRNN(
            VOCAB_SIZE, cell=cell, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE
        )

    def test_appends_tokens_after_prompt(self) -> None:
        """Generation keeps the prompt, adds valid ids and restores the mode."""
        prompt = torch.tensor([1, 2])
        for cell in CELL_TYPES:
            with self.subTest(cell=cell):
                model = self.build(cell).train()
                out = model.generate(prompt, max_new_tokens=10)
                self.assertEqual(out.shape, (12,))
                self.assertTrue(torch.equal(out[:2], prompt))
                self.assertTrue(((out >= 0) & (out < VOCAB_SIZE)).all())
                self.assertTrue(model.training)

    def test_single_token_prompt(self) -> None:
        """Unlike CharMLP, a single token is enough to start generating."""
        out = self.build("gru").generate(torch.tensor([0]), max_new_tokens=5)
        self.assertEqual(out.shape, (6,))

    def test_empty_prompt_raises(self) -> None:
        """The prompt needs at least one token to start the state."""
        with self.assertRaises(ValueError):
            self.build("rnn").generate(torch.tensor([], dtype=torch.long), 5)

    def test_reproducible_with_generator(self) -> None:
        """Equally seeded generators yield the same text."""
        model = self.build("lstm")
        prompt = torch.tensor([0])
        outs = [
            model.generate(prompt, 20, generator=torch.Generator().manual_seed(3))
            for _ in range(2)
        ]
        self.assertTrue(torch.equal(outs[0], outs[1]))


if __name__ == "__main__":
    unittest.main()
