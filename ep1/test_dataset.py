"""Tests for the generic dataset pipeline utilities."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import transforms

from ep1.dataset import get_pipeline, split_dataset

# A 2x2 grayscale image with known pixel values, used as a didactic example.
TOY_PIXELS = [0, 255, 0, 255]
TOY_MEAN = (0.5,)
TOY_STD = (0.5,)

# A tiny labelled dataset, large enough to split but small enough to reason about.
TOY_DATASET_SIZE = 10


def make_toy_image() -> Image.Image:
    """Builds a tiny 2x2 grayscale image with pixels 0 and 255."""
    image = Image.new("L", (2, 2))
    image.putdata(TOY_PIXELS)
    return image


def make_toy_dataset() -> TensorDataset:
    """Builds a 10-sample dataset whose features and labels are 0..9."""
    values = torch.arange(TOY_DATASET_SIZE)
    return TensorDataset(values.float().unsqueeze(1), values)


class TestGetPipeline(unittest.TestCase):
    """Tests for get_pipeline."""

    def test_without_normalization_only_converts_to_tensor(self) -> None:
        """Without mean/std the pipeline holds a single ToTensor step."""
        pipeline = get_pipeline()

        self.assertEqual(len(pipeline.transforms), 1)
        self.assertIsInstance(pipeline.transforms[0], transforms.ToTensor)

    def test_with_normalization_appends_normalize(self) -> None:
        """Passing mean/std appends a Normalize step with those values."""
        pipeline = get_pipeline(mean=TOY_MEAN, std=TOY_STD)
        normalize = pipeline.transforms[1]

        self.assertEqual(len(pipeline.transforms), 2)
        self.assertIsInstance(normalize, transforms.Normalize)
        self.assertEqual(tuple(normalize.mean), TOY_MEAN)
        self.assertEqual(tuple(normalize.std), TOY_STD)

    def test_to_tensor_scales_pixels_to_unit_range(self) -> None:
        """ToTensor maps pixels from [0, 255] to [0.0, 1.0] as a 1x2x2 tensor."""
        pipeline = get_pipeline()
        tensor = pipeline(make_toy_image())

        self.assertEqual(tensor.shape, (1, 2, 2))
        self.assertEqual(tensor.flatten().tolist(), [0.0, 1.0, 0.0, 1.0])

    def test_normalization_centers_pixels(self) -> None:
        """With mean=0.5 and std=0.5 the unit range is mapped to [-1.0, 1.0]."""
        pipeline = get_pipeline(mean=TOY_MEAN, std=TOY_STD)
        tensor = pipeline(make_toy_image())

        self.assertEqual(tensor.flatten().tolist(), [-1.0, 1.0, -1.0, 1.0])


class TestSplitDataset(unittest.TestCase):
    """Tests for split_dataset."""

    def setUp(self) -> None:
        """Creates the 10-sample toy dataset shared by these tests."""
        self.dataset = make_toy_dataset()

    def test_split_by_count_gives_requested_sizes(self) -> None:
        """An int val_split holds out exactly that many samples."""
        train, val = split_dataset(self.dataset, 3)

        self.assertEqual(len(train), 7)
        self.assertEqual(len(val), 3)

    def test_split_by_fraction_gives_proportional_sizes(self) -> None:
        """A float val_split holds out that fraction of the dataset."""
        train, val = split_dataset(self.dataset, 0.2)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 2)

    def test_split_is_a_partition_of_the_dataset(self) -> None:
        """Every sample lands in exactly one of the two subsets."""
        train, val = split_dataset(self.dataset, 3)

        indices = sorted(train.indices + val.indices)

        self.assertEqual(indices, list(range(TOY_DATASET_SIZE)))

    def test_zero_val_split_gives_empty_validation_set(self) -> None:
        """A val_split of 0 is legal and expresses "no validation"."""
        train, val = split_dataset(self.dataset, 0)

        self.assertEqual(len(train), TOY_DATASET_SIZE)
        self.assertEqual(len(val), 0)

    def test_same_seed_gives_identical_indices(self) -> None:
        """Two splits with the same seed select the same samples."""
        first_train, first_val = split_dataset(self.dataset, 3, seed=7)
        second_train, second_val = split_dataset(self.dataset, 3, seed=7)

        self.assertEqual(first_train.indices, second_train.indices)
        self.assertEqual(first_val.indices, second_val.indices)

    def test_negative_count_raises(self) -> None:
        """A negative sample count is rejected instead of being clamped."""
        with self.assertRaises(ValueError):
            split_dataset(self.dataset, -1)

    def test_negative_fraction_raises(self) -> None:
        """A negative fraction is rejected instead of being clamped."""
        with self.assertRaises(ValueError):
            split_dataset(self.dataset, -0.1)

    def test_count_larger_than_dataset_raises(self) -> None:
        """Holding out more samples than exist is rejected."""
        with self.assertRaises(ValueError):
            split_dataset(self.dataset, TOY_DATASET_SIZE + 1)

    def test_count_equal_to_dataset_raises(self) -> None:
        """Holding out every sample would leave no training data."""
        with self.assertRaises(ValueError):
            split_dataset(self.dataset, TOY_DATASET_SIZE)

    def test_fraction_of_one_raises(self) -> None:
        """A fraction of 1.0 would leave no training data."""
        with self.assertRaises(ValueError):
            split_dataset(self.dataset, 1.0)

    def test_fraction_above_one_raises(self) -> None:
        """A fraction above 1.0 is rejected instead of being clamped."""
        with self.assertRaises(ValueError):
            split_dataset(self.dataset, 1.5)


if __name__ == "__main__":
    unittest.main()
