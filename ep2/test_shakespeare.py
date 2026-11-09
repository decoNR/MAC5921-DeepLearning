"""Tests for the Tiny Shakespeare experiment script."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ep1.models import count_parameters
from ep2.data import build_corpus
from ep2.shakespeare import (
    PROMPT_LENGTH,
    ModelSpec,
    batch_size_for,
    build_model,
    hidden_size_for_budget,
    parameter_count,
    parse_args,
    parse_model_spec,
    run_experiment,
)

VOCAB_SIZE = 65
EMBED_DIM = 24
SPECS = [ModelSpec("mlp", 1), ModelSpec("mlp", 8), ModelSpec("rnn"), ModelSpec("gru"), ModelSpec("lstm")]


class TestParseModelSpec(unittest.TestCase):
    """Tests for the model spec parser."""

    def test_valid_specs(self) -> None:
        """MLP specs carry their window; recurrent specs carry none."""
        self.assertEqual(parse_model_spec("mlp:8"), ModelSpec("mlp", 8))
        self.assertEqual(parse_model_spec("LSTM"), ModelSpec("lstm"))
        self.assertEqual(parse_model_spec("gru").name, "GRU")
        self.assertEqual(parse_model_spec("mlp:3").name, "MLP (k=3)")

    def test_invalid_specs(self) -> None:
        """Missing or invalid windows and unknown kinds are rejected."""
        for text in ("mlp", "mlp:0", "mlp:x", f"mlp:{PROMPT_LENGTH + 1}", "lstm:3", "cnn"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_model_spec(text)


class TestParameterBudget(unittest.TestCase):
    """Tests for the parameter formulas and the budget search."""

    def test_formulas_match_built_models(self) -> None:
        """The analytic counts equal count_parameters on the real models."""
        for spec in SPECS:
            for hidden in (7, 64):
                with self.subTest(spec=spec, hidden=hidden):
                    model = build_model(spec, VOCAB_SIZE, EMBED_DIM, hidden)
                    self.assertEqual(
                        parameter_count(spec, VOCAB_SIZE, EMBED_DIM, hidden),
                        count_parameters(model),
                    )

    def test_hidden_size_is_closest_to_budget(self) -> None:
        """No neighbouring hidden size lands closer to the budget."""
        budget = 150_000
        for spec in SPECS:
            with self.subTest(spec=spec):
                hidden = hidden_size_for_budget(spec, VOCAB_SIZE, EMBED_DIM, budget)
                distance = abs(parameter_count(spec, VOCAB_SIZE, EMBED_DIM, hidden) - budget)
                for neighbour in (hidden - 1, hidden + 1):
                    other = abs(parameter_count(spec, VOCAB_SIZE, EMBED_DIM, neighbour) - budget)
                    self.assertLessEqual(distance, other)

    def test_all_models_within_two_percent(self) -> None:
        """At the default budget every model is within 2% of it."""
        budget = 150_000
        for spec in SPECS:
            with self.subTest(spec=spec):
                hidden = hidden_size_for_budget(spec, VOCAB_SIZE, EMBED_DIM, budget)
                count = parameter_count(spec, VOCAB_SIZE, EMBED_DIM, hidden)
                self.assertLess(abs(count - budget) / budget, 0.02)


class TestBatchSize(unittest.TestCase):
    """Tests for the equal-tokens-per-step batch sizes."""

    def test_same_predictions_per_step(self) -> None:
        """An MLP batch and an RNN batch hold the same number of predictions."""
        mlp = batch_size_for(ModelSpec("mlp", 8), 3200, 100)
        rnn = batch_size_for(ModelSpec("lstm"), 3200, 100)
        self.assertEqual(mlp * 1, rnn * 100)

    def test_tokens_must_be_multiple_of_seq_len(self) -> None:
        """RNN batches must be whole sequences."""
        with self.assertRaises(ValueError):
            batch_size_for(ModelSpec("rnn"), 3250, 100)


class TestParseArgs(unittest.TestCase):
    """Tests for the command-line interface."""

    def test_defaults_and_overrides(self) -> None:
        """Arguments default sensibly and can be overridden."""
        args = parse_args(["--models", "mlp:4", "gru", "--epochs", "2", "--grad-clip", "0"])
        self.assertEqual(args.models, ["mlp:4", "gru"])
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.grad_clip, 0.0)
        self.assertEqual(parse_args([]).name, "baseline")


class TestRunExperiment(unittest.TestCase):
    """End-to-end test on a tiny corpus."""

    def test_writes_results(self) -> None:
        """A one-epoch run trains every model and writes all result files."""
        text = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 30
        corpus = build_corpus(text)
        specs = [ModelSpec("mlp", 4), ModelSpec("gru")]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                summaries = run_experiment(
                    model_specs=specs, epochs=1, budget=5_000, seq_len=10,
                    tokens_per_batch=50, sample_length=20, corpus=corpus,
                    results_dir=out, device=torch.device("cpu"),
                )
            self.assertEqual(set(summaries), {"MLP (k=4)", "GRU"})
            for name in ("curves.png", "history.json", "summary.json", "samples.txt"):
                self.assertTrue((out / name).exists(), name)
            saved = json.loads((out / "summary.json").read_text())
            self.assertEqual(saved["GRU"]["best_epoch"], 1)
            self.assertIn("MLP (k=4)", (out / "samples.txt").read_text())


if __name__ == "__main__":
    unittest.main()
