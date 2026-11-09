"""Training and evaluation loop for the character-level language models."""

import copy
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from library.dataset import DEFAULT_SEED
from library.train import get_device, set_seed

DEFAULT_GRAD_CLIP = 1.0


def _batch_loss(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, int, int]:
    """Returns the mean loss.

    Flattening makes the MLP's (B, V) and the RNN's (B, T, V) logits look the
    same: one row per predicted character.
    """
    logits = model(inputs)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_targets)
    correct = (flat_logits.argmax(dim=-1) == flat_targets).sum().item()
    return loss, correct, flat_targets.numel()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device],
    optimizer: Optional[Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float]:
    """Runs one pass over the loader and returns the mean loss and accuracy.

    With an optimizer the weights are updated after every batch; without one
    the pass only measures, with gradients disabled. Averages are weighted by
    the number of predicted characters, so a smaller last batch counts less.
    """
    if device is None:
        device = get_device()
    training = optimizer is not None
    was_training = model.training
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()
            loss, correct, count = _batch_loss(model, inputs, targets)
            if training:
                loss.backward()
                if grad_clip is not None:
                    # Rescales the whole gradient when its norm exceeds grad_clip, which
                    # prevents the exploding gradients that recurrent networks are prone to.
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * count
            total_correct += correct
            total_count += count

    model.train(was_training)
    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: Optional[torch.device] = None,
    grad_clip: Optional[float] = DEFAULT_GRAD_CLIP,
) -> Tuple[float, float]:
    """Trains for one epoch and returns the mean loss and accuracy per character."""
    return _run_epoch(model, loader, device, optimizer, grad_clip)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Returns the mean loss and accuracy per character, without updating weights."""
    return _run_epoch(model, loader, device)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    grad_clip: Optional[float] = DEFAULT_GRAD_CLIP,
    patience: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[float]]:
    """Trains with Adam and returns the per-epoch history.

    After training, the model is restored to the weights of the epoch with the
    lowest validation loss, so later test evaluation uses the selected model.
    """
    set_seed(seed)
    if device is None:
        device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }
    best_val_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement in {patience} epochs.")
            break

    model.load_state_dict(best_state)
    return history
