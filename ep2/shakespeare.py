"""Compares an MLP (fixed window) with RNN/GRU/LSTM models on Tiny Shakespeare.

Every model gets:

* the same parameter budget: its hidden size is chosen so that its number of
  trainable parameters is as close as possible to `--budget`;
* the same training budget: the same epochs, the same number of optimizer
  steps per epoch and the same number of predicted characters per step. An MLP
  batch holds `tokens_per_batch` windows (one prediction each) and an RNN
  batch `tokens_per_batch / seq_len` sequences (seq_len predictions each).

Usage:
    python ep2/shakespeare.py
    python ep2/shakespeare.py --models mlp:1 mlp:8 lstm --epochs 5 --name quick
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from ep1.models import count_parameters
from ep2.data import (
    TextCorpus,
    build_datasets,
    create_text_dataloaders,
    load_tiny_shakespeare,
)
from ep2.models import CELL_TYPES, GATES_PER_CELL, CharMLP, CharRNN
from ep2.train import DEFAULT_GRAD_CLIP, evaluate, train_model
from library.dataset import DEFAULT_SEED
from library.train import get_device, set_seed
from library.visualization import plot_curves

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_MODELS = ("mlp:1", "mlp:3", "mlp:8", "mlp:16", "mlp:32", "rnn", "gru", "lstm")
EMBED_DIM = 24
PARAMETER_BUDGET = 150_000
SEQ_LEN = 100
TOKENS_PER_BATCH = 3_200
EPOCHS = 10
LEARNING_RATE = 2e-3
PROMPT_LENGTH = 64
SAMPLE_LENGTH = 400
SAMPLE_TEMPERATURE = 0.8


@dataclass(frozen=True)
class ModelSpec:
    """A model family ("mlp", "rnn", "gru" or "lstm") and, for the MLP, its window."""

    kind: str
    context: Optional[int] = None

    @property
    def name(self) -> str:
        """Returns a readable name, e.g. "MLP (k=8)" or "LSTM"."""
        if self.kind == "mlp":
            return f"MLP (k={self.context})"
        return self.kind.upper()


def parse_model_spec(text: str) -> ModelSpec:
    """Parses "mlp:<k>", "rnn", "gru" or "lstm"."""
    kind, _, context = text.lower().partition(":")
    if kind == "mlp":
        if not context.isdigit() or int(context) < 1:
            raise ValueError(f"MLP spec must look like 'mlp:<k>' with k >= 1, got {text!r}")
        if int(context) > PROMPT_LENGTH:
            raise ValueError(f"MLP window must be <= {PROMPT_LENGTH}, got {context}")
        return ModelSpec("mlp", int(context))
    if kind in CELL_TYPES and not context:
        return ModelSpec(kind)
    raise ValueError(f"Unknown model spec {text!r}; use mlp:<k>, rnn, gru or lstm")


def mlp_parameter_count(vocab_size: int, context: int, embed_dim: int, hidden: int) -> int:
    """Embedding V*E + hidden layer (k*E + 1)*H + output layer (H + 1)*V."""
    return (
        vocab_size * embed_dim
        + (context * embed_dim + 1) * hidden
        + (hidden + 1) * vocab_size
    )


def rnn_parameter_count(vocab_size: int, cell: str, embed_dim: int, hidden: int) -> int:
    """Embedding V*E + gates*(E*H + H*H + 2H) + output layer (H + 1)*V (one layer)."""
    recurrent = GATES_PER_CELL[cell] * (embed_dim * hidden + hidden * hidden + 2 * hidden)
    return vocab_size * embed_dim + recurrent + (hidden + 1) * vocab_size


def parameter_count(spec: ModelSpec, vocab_size: int, embed_dim: int, hidden: int) -> int:
    """Returns the number of trainable parameters of a model without building it."""
    if spec.kind == "mlp":
        return mlp_parameter_count(vocab_size, spec.context, embed_dim, hidden)
    return rnn_parameter_count(vocab_size, spec.kind, embed_dim, hidden)


def hidden_size_for_budget(
    spec: ModelSpec, vocab_size: int, embed_dim: int, budget: int
) -> int:
    """Returns the hidden size whose parameter count is closest to the budget."""
    low, high = 1, 1
    while parameter_count(spec, vocab_size, embed_dim, high) <= budget:
        high *= 2
    while low < high:
        mid = (low + high + 1) // 2
        if parameter_count(spec, vocab_size, embed_dim, mid) <= budget:
            low = mid
        else:
            high = mid - 1
    below = parameter_count(spec, vocab_size, embed_dim, low)
    above = parameter_count(spec, vocab_size, embed_dim, low + 1)
    return low + 1 if above - budget < budget - below else low


def build_model(spec: ModelSpec, vocab_size: int, embed_dim: int, hidden: int) -> nn.Module:
    """Instantiates the model described by spec."""
    if spec.kind == "mlp":
        return CharMLP(vocab_size, spec.context, embed_dim=embed_dim, hidden_size=hidden)
    return CharRNN(vocab_size, cell=spec.kind, embed_dim=embed_dim, hidden_size=hidden)


def batch_size_for(spec: ModelSpec, tokens_per_batch: int, seq_len: int) -> int:
    """Returns the batch size giving tokens_per_batch predictions per step."""
    if spec.kind == "mlp":
        return tokens_per_batch
    if tokens_per_batch % seq_len != 0:
        raise ValueError(
            f"tokens_per_batch ({tokens_per_batch}) must be a multiple of "
            f"seq_len ({seq_len})"
        )
    return tokens_per_batch // seq_len


def run_experiment(
    experiment_name: str = "baseline",
    model_specs: Sequence[ModelSpec] = tuple(parse_model_spec(m) for m in DEFAULT_MODELS),
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    budget: int = PARAMETER_BUDGET,
    embed_dim: int = EMBED_DIM,
    seq_len: int = SEQ_LEN,
    tokens_per_batch: int = TOKENS_PER_BATCH,
    grad_clip: Optional[float] = DEFAULT_GRAD_CLIP,
    sample_length: int = SAMPLE_LENGTH,
    seed: int = DEFAULT_SEED,
    corpus: Optional[TextCorpus] = None,
    results_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, Any]]:
    """Trains every model, evaluates it on the test split and saves the results.

    Writes curves.png, history.json, summary.json and samples.txt to
    results_dir (ep2/results/<experiment_name> by default).
    """
    if corpus is None:
        corpus = load_tiny_shakespeare()
    if results_dir is None:
        results_dir = RESULTS_DIR / experiment_name
    if device is None:
        device = get_device()
    results_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = corpus.vocab.size
    prompt = corpus.test_data[:PROMPT_LENGTH]
    print(f"Using device: {device} | Vocabulary: {vocab_size} characters")

    histories: Dict[str, Dict[str, List[float]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    samples: Dict[str, str] = {}

    for spec in model_specs:
        hidden = hidden_size_for_budget(spec, vocab_size, embed_dim, budget)
        set_seed(seed)
        model = build_model(spec, vocab_size, embed_dim, hidden)
        num_params = count_parameters(model)
        print(f"\n{'=' * 20} {spec.name}: hidden={hidden}, params={num_params:,} {'=' * 20}")

        mode, length = ("next_char", spec.context) if spec.kind == "mlp" else ("sequence", seq_len)
        train_loader, val_loader, test_loader = create_text_dataloaders(
            *build_datasets(corpus, mode, length),
            batch_size=batch_size_for(spec, tokens_per_batch, seq_len),
            seed=seed,
        )
        history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, grad_clip=grad_clip, device=device, seed=seed,
        )
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"{spec.name} Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        generated = model.generate(
            prompt.to(device), sample_length, temperature=SAMPLE_TEMPERATURE,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        samples[spec.name] = corpus.vocab.decode(generated[len(prompt):])

        best_epoch = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)
        histories[spec.name] = history
        summaries[spec.name] = {
            "model_name": spec.name,
            "hidden_size": hidden,
            "num_params": num_params,
            "best_val_loss": history["val_loss"][best_epoch],
            "best_epoch": best_epoch + 1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "mean_epoch_time": sum(history["epoch_time"]) / len(history["epoch_time"]),
        }

    print("\n" + format_table(summaries.values()))
    plot_curves(histories, out_path=results_dir / "curves.png")
    (results_dir / "history.json").write_text(json.dumps(histories, indent=2))
    (results_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    (results_dir / "samples.txt").write_text(
        format_samples(corpus.vocab.decode(prompt), samples), encoding="utf-8"
    )
    return summaries


def format_table(summaries: Sequence[Dict[str, Any]]) -> str:
    """Formats the summaries as a markdown table, one row per model."""
    lines = [
        "| Model | Hidden | Params | Best Val Loss | Best Epoch | Test Loss | Test Acc | Time/Epoch (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model_name']} | {s['hidden_size']} | {s['num_params']:,} | "
            f"{s['best_val_loss']:.4f} | {s['best_epoch']} | {s['test_loss']:.4f} | "
            f"{s['test_acc']:.4f} | {s['mean_epoch_time']:.1f} |"
        )
    return "\n".join(lines)


def format_samples(prompt: str, samples: Dict[str, str]) -> str:
    """Formats the generated continuations of the shared prompt."""
    parts = [f"Prompt (temperature {SAMPLE_TEMPERATURE}):\n{prompt}\n"]
    for name, text in samples.items():
        parts.append(f"{'=' * 20} {name} {'=' * 20}\n{text}\n")
    return "\n".join(parts)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="baseline", help="Results subdirectory name.")
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS),
        help="Models to compare: mlp:<k>, rnn, gru and/or lstm.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--budget", type=int, default=PARAMETER_BUDGET,
        help="Target number of trainable parameters for every model.",
    )
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="RNN sequence length.")
    parser.add_argument(
        "--tokens-per-batch", type=int, default=TOKENS_PER_BATCH,
        help="Predicted characters per optimizer step, for every model.",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=DEFAULT_GRAD_CLIP,
        help="Maximum gradient norm; 0 disables clipping.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main() -> None:
    """Runs the comparison from the command line."""
    args = parse_args()
    run_experiment(
        experiment_name=args.name,
        model_specs=[parse_model_spec(m) for m in args.models],
        epochs=args.epochs,
        lr=args.lr,
        budget=args.budget,
        seq_len=args.seq_len,
        tokens_per_batch=args.tokens_per_batch,
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
