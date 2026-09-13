"""Generic dataset pipeline utilities."""

from typing import Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, random_split
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

