"""Experiment script comparing MLP and CNN on the MNIST dataset."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torchvision import datasets

from ep1.dataset import DEFAULT_SEED, create_dataloaders, get_pipeline
from ep1.models import count_parameters, make_cnn, make_mlp
from ep1.report import plot_curves, print_comparison
from ep1.train import evaluate, get_device, set_seed, summarize, train_model

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

CURVES_PATH = RESULTS_DIR / "curves.png"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
HISTORY_PATH = RESULTS_DIR / "history.json"

# Holds out 12,000 of the 60,000 training images for validation.
VAL_SPLIT = 0.2

EPOCHS = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
SEED = DEFAULT_SEED


def run_experiment(
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    results_dir: Path = RESULTS_DIR,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, Any]]]:
    """Executes the comparative experiment between MLP and CNN on MNIST."""
    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    set_seed(seed)

    pipeline = get_pipeline(mean=MNIST_MEAN, std=MNIST_STD)
    training_data = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=pipeline
    )
    test_data = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=pipeline
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        training_data,
        test_dataset=test_data,
        val_split=VAL_SPLIT,
        batch_size=batch_size,
        seed=seed,
    )
    assert test_loader is not None, "test_loader must not be None"

    models_to_train = [
        ("MLP", make_mlp),
        ("CNN", make_cnn),
    ]

    histories: Dict[str, Dict[str, List[float]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    criterion = nn.CrossEntropyLoss()

    for model_name, model_fn in models_to_train:
        print(f"\n{'=' * 20} Training {model_name} {'=' * 20}")
        set_seed(seed)
        model = model_fn().to(device)
        num_params = count_parameters(model)
        print(f"{model_name} trainable parameters: {num_params:,}")

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            seed=seed,
        )
        histories[model_name] = history

        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"{model_name} Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        summary = summarize(
            history=history,
            model_name=model_name,
            num_params=num_params,
            test_acc=test_acc,
        )
        summaries[model_name] = summary

    print("\n" + "=" * 20 + " Comparison " + "=" * 20)
    results_dir.mkdir(parents=True, exist_ok=True)
    print_comparison(summaries, out_path=results_dir / "summary.json")
    plot_curves(histories, out_path=results_dir / "curves.png")

    with open(results_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)
    print(f"\nExperiment complete. Results saved to {results_dir}")

    return histories, summaries


def main() -> None:
    """Runs the training and comparison experiment between MLP and CNN on MNIST."""
    run_experiment()


if __name__ == "__main__":
    main()
