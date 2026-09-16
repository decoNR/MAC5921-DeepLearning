"""Training and evaluation routines for model comparison."""

import random
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ep1.dataset import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Sets the random seed for Python and PyTorch to ensure reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Returns a CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Trains the model for one epoch over the given loader."""
    if device is None:
        device = get_device()

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=-1)
        total_correct += (predictions == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Evaluates the model without computing gradients."""
    if device is None:
        device = get_device()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy

