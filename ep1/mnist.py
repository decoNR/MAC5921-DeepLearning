"""MNIST work built on top of the generic dataset pipeline utilities."""

import sys
from pathlib import Path

# Ensure the project root is importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ep1.dataset import get_pipeline

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def main() -> None:
    """Builds the MNIST transform pipeline."""
    pipeline = get_pipeline(mean=MNIST_MEAN, std=MNIST_STD)

    print(f"MNIST pipeline: {pipeline}")


if __name__ == "__main__":
    main()
