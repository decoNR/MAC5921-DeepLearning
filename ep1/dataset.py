"""Generic dataset pipeline utilities."""

from typing import Optional, Sequence

from torchvision import transforms


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
