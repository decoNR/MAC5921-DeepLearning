import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets

from ep1.dataset import (
    DEFAULT_SEED,
    AddGaussianNoise,
    PerSampleSeededDataset,
    create_dataloaders,
    get_pipeline,
)
from ep1.models import count_parameters, make_cnn, make_mlp
from ep1.visualization import plot_curves, print_comparison, save_sample_images
from ep1.train import evaluate, get_device, set_seed, summarize, train_model

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "ep1" / "results"

# Holds out 12,000 of the 60,000 training images for validation.
VAL_SPLIT = 0.2

EPOCHS = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
SEED = DEFAULT_SEED

VALID_EXPERIMENTS = (
    "baseline",
    "noise_gaussian_std0.2",
    "test_noise_gaussian_std0.2",
    "test_noise_gaussian_std0.3",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Data transformations that define an experiment."""

    # Applied to the training, validation and test sets alike.
    pre_transforms: Optional[Sequence[Any]] = None
    post_transforms: Optional[Sequence[Any]] = None
    # Applied only to the test set, after the shared transforms above.
    test_pre_transforms: Optional[Sequence[Any]] = None
    test_post_transforms: Optional[Sequence[Any]] = None


def get_experiment_config(experiment_name: str) -> ExperimentConfig:
    """Returns the configuration associated with an experiment name."""
    if experiment_name == "baseline":
        return ExperimentConfig()
    elif experiment_name == "noise_gaussian_std0.2":
        return ExperimentConfig(post_transforms=[AddGaussianNoise(mean=0.0, std=0.2)])
    elif experiment_name == "test_noise_gaussian_std0.2":
        return ExperimentConfig(
            test_post_transforms=[AddGaussianNoise(mean=0.0, std=0.2)]
        )
    elif experiment_name == "test_noise_gaussian_std0.3":
        return ExperimentConfig(
            test_post_transforms=[AddGaussianNoise(mean=0.0, std=0.3)]
        )
    else:
        raise ValueError(
            f"Invalid experiment_name '{experiment_name}'. "
            f"Must be one of: {list(VALID_EXPERIMENTS)}"
        )


def build_test_dataset(config: ExperimentConfig, seed: int) -> Dataset:
    """Builds the MNIST test set with the shared and the test-only transforms."""
    pipeline = get_pipeline(
        mean=MNIST_MEAN,
        std=MNIST_STD,
        pre_transforms=[
            *(config.pre_transforms or []),
            *(config.test_pre_transforms or []),
        ],
        post_transforms=[
            *(config.post_transforms or []),
            *(config.test_post_transforms or []),
        ],
    )
    test_data = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=pipeline
    )
    return PerSampleSeededDataset(test_data, seed=seed)


def has_test_only_transforms(config: ExperimentConfig) -> bool:
    """Tells whether the test set is transformed differently from training."""
    return bool(config.test_pre_transforms or config.test_post_transforms)


def run_experiment(
    experiment_name: str = "baseline",
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    results_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, Any]]]:
    """Executes the comparative experiment between MLP and CNN on MNIST."""
    config = get_experiment_config(experiment_name)

    if device is None:
        device = get_device()
    print(f"Using device: {device}")
    print(f"Running experiment: '{experiment_name}'")

    if results_dir is None:
        results_dir = RESULTS_DIR / experiment_name

    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    pipeline = get_pipeline(
        mean=MNIST_MEAN,
        std=MNIST_STD,
        pre_transforms=config.pre_transforms,
        post_transforms=config.post_transforms,
    )
    training_data = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=pipeline
    )
    test_data = build_test_dataset(config, seed)

    save_sample_images(
        training_data,
        out_path=results_dir / "samples.png",
        num_samples=8,
        title=f"MNIST Training Samples: {experiment_name}",
    )
    if has_test_only_transforms(config):
        save_sample_images(
            test_data,
            out_path=results_dir / "samples_test.png",
            num_samples=8,
            title=f"MNIST Test Samples: {experiment_name}",
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
    print_comparison(summaries, out_path=results_dir / "summary.json")
    plot_curves(histories, out_path=results_dir / "curves.png")

    with open(results_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)
    print(f"\nExperiment complete. Results saved to {results_dir}")

    return histories, summaries


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Run comparative MLP vs CNN experiments on MNIST."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        help=f"Experiment to run. Must be one of: {list(VALID_EXPERIMENTS)}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED}).",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the training and comparison experiment between MLP and CNN on MNIST."""
    args = parse_args()

    run_experiment(
        experiment_name=args.experiment,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
