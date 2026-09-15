"""Training and evaluation routines for model comparison."""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ep1.dataset import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Sets the random seed for Python and PyTorch to ensure reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Returns a CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
