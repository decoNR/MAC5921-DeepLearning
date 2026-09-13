"""MNIST work built on top of the generic dataset pipeline utilities."""

import sys
from pathlib import Path

# Ensure the project root is importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torchvision import datasets

from ep1.dataset import get_pipeline, split_dataset

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


DATA_DIR = PROJECT_ROOT / "data"

# Holds out 12,000 of the 60,000 training images for validation.
VAL_SPLIT = 0.2


def main() -> None:
    """Builds the MNIST pipeline and splits the training set for validation."""
    pipeline = get_pipeline(mean=MNIST_MEAN, std=MNIST_STD)
    training_data = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=pipeline
    )
    train_data, val_data = split_dataset(training_data, VAL_SPLIT)

    print(f"MNIST pipeline: {pipeline}")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")


if __name__ == "__main__":
    main()
