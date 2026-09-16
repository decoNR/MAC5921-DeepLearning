"""Training and evaluation routines for model comparison."""

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 15,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[float]]:
    """Trains a model across epochs using Adam and CrossEntropyLoss."""
    set_seed(seed)

    if device is None:
        device = get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            elapsed = time.perf_counter() - start_time
            history["epoch_time"].append(elapsed)
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Time: {elapsed:.2f}s"
            )
        else:
            elapsed = time.perf_counter() - start_time
            history["epoch_time"].append(elapsed)
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

    return history


def summarize(
    history: Dict[str, List[float]],
    model_name: str,
    num_params: int,
    test_acc: float,
) -> Dict[str, Any]:
    """Summarizes training metrics into a serializable dictionary."""
    val_acc = history.get("val_acc", [])
    if val_acc:
        best_val_acc = max(val_acc)
        best_val_epoch = val_acc.index(best_val_acc) + 1
        threshold = 0.99 * best_val_acc
        epochs_to_convergence = next(
            (i + 1 for i, acc in enumerate(val_acc) if acc >= threshold),
            best_val_epoch,
        )
    else:
        best_val_acc = 0.0
        best_val_epoch = 0
        epochs_to_convergence = 0

    epoch_times = history.get("epoch_time", [])
    total_time = sum(epoch_times)
    mean_epoch_time = total_time / len(epoch_times) if epoch_times else 0.0

    return {
        "model_name": model_name,
        "num_params": int(num_params),
        "best_val_acc": float(best_val_acc),
        "best_val_epoch": int(best_val_epoch),
        "epochs_to_convergence": int(epochs_to_convergence),
        "mean_epoch_time": float(mean_epoch_time),
        "total_time": float(total_time),
        "test_acc": float(test_acc),
    }

