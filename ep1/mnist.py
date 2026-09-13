"""MNIST work built on top of the generic dataset pipeline utilities."""

import sys
from pathlib import Path

# Ensure the project root is importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torchvision import datasets

from ep1.dataset import create_dataloaders, get_pipeline

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


DATA_DIR = PROJECT_ROOT / "data"

# Holds out 12,000 of the 60,000 training images for validation.
VAL_SPLIT = 0.2

BATCH_SIZE = 64


def main() -> None:
    """Builds the MNIST loaders for training, validation and testing."""
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
        batch_size=BATCH_SIZE,
    )

    print(f"MNIST pipeline: {pipeline}")
    for name, loader in (
        ("Train", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader),
    ):
        print(
            f"{name}: {len(loader.dataset)} samples in {len(loader)} batches "
            f"of up to {loader.batch_size}"
        )


if __name__ == "__main__":
    main()
