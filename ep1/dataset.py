"""Generic dataset pipeline utilities."""

from typing import Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

DEFAULT_SEED = 44


def get_pipeline(
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> transforms.Compose:
    """Builds the transform pipeline applied to every sample.

    Converts inputs to tensors and, when mean and std are given, normalizes them.
    """
    ops = [transforms.ToTensor()]
    if mean is not None and std is not None:
        ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)


def split_dataset(
    dataset: Dataset,
    val_split: Union[int, float],
    seed: int = DEFAULT_SEED,
) -> Tuple[Dataset, Dataset]:
    """Splits a dataset into a training and a validation subset."""
    size = len(dataset)
    if isinstance(val_split, bool):
        raise TypeError("val_split must be an int or a float, got a bool")

    if isinstance(val_split, int):
        val_size = val_split
        if not 0 <= val_size < size:
            raise ValueError(
                f"val_split must be in [0, {size}) samples, got {val_split}"
            )
    elif isinstance(val_split, float):
        if not 0.0 <= val_split < 1.0:
            raise ValueError(f"val_split must be in [0.0, 1.0), got {val_split}")
        val_size = int(val_split * size)
    else:
        raise TypeError(
            f"val_split must be an int or a float, got {type(val_split).__name__}"
        )

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        dataset, [size - val_size, val_size], generator=generator
    )

    return train_subset, val_subset


def _check_batch_size(name: str, value: int) -> None:
    """Rejects batch sizes that are not strictly positive integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


def _make_eval_loader(
    dataset: Dataset, batch_size: int, num_workers: int, seed: int
) -> DataLoader:
    """Builds a loader for evaluation data, which is never shuffled."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # Without its own generator a loader draws from the global RNG, so
        # evaluating could shift training's dropout and augmentation draws.
        generator=torch.Generator().manual_seed(seed),
    )


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    val_split: Union[int, float] = 0,
    batch_size: int = 64,
    eval_batch_size: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Wraps datasets into loaders for training, validation and testing."""
    if eval_batch_size is None:
        eval_batch_size = batch_size

    _check_batch_size("batch_size", batch_size)
    _check_batch_size("eval_batch_size", eval_batch_size)

    val_dataset = None
    if val_split > 0:
        train_dataset, val_dataset = split_dataset(train_dataset, val_split, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # Seeding the shuffle keeps a whole run reproducible, not just the split.
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = _make_eval_loader(
            val_dataset, eval_batch_size, num_workers, seed
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = _make_eval_loader(
            test_dataset, eval_batch_size, num_workers, seed
        )

    return train_loader, val_loader, test_loader

