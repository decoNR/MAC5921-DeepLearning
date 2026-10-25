"""Tests for the generic dataset pipeline utilities."""

import sys
import unittest
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from torchvision import transforms

from ep1.dataset import (
    AddGaussianNoise,
    PerSampleSeededDataset,
    create_dataloaders,
    get_pipeline,
    split_dataset,
)

# A 2x2 grayscale image with known pixel values, used as a didactic example.
TOY_PIXELS = [0, 255, 0, 255]
TOY_MEAN = (0.5,)
TOY_STD = (0.5,)

# A tiny labelled dataset, large enough to split but small enough to reason about.
TOY_DATASET_SIZE = 10

# A separate, differently sized stand-in for a held-out test set.
TOY_TEST_DATASET_SIZE = 4


def make_toy_image() -> Image.Image:
    """Builds a tiny 2x2 grayscale image with pixels 0 and 255."""
    image = Image.new("L", (2, 2))
    image.putdata(TOY_PIXELS)
    return image


def make_toy_dataset(size: int = TOY_DATASET_SIZE) -> TensorDataset:
    """Builds a dataset whose features and labels count up from 0."""
    values = torch.arange(size)
    return TensorDataset(values.float().unsqueeze(1), values)


def collect_labels(loader: DataLoader) -> List[int]:
    """Reads every label a loader yields, in the order it yields them."""
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.tolist())
    return labels


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

    def test_pipeline_with_pre_and_post_transforms(self) -> None:
        """Pre-transforms run before ToTensor, post-transforms run after."""
        class MockPre:
            def __call__(self, img: Image.Image) -> Image.Image:
                return img

        class MockPost:
            def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
                return tensor * 2.0

        pipeline = get_pipeline(
            pre_transforms=[MockPre()],
            post_transforms=[MockPost()],
        )
        self.assertEqual(len(pipeline.transforms), 3)
        self.assertIsInstance(pipeline.transforms[0], MockPre)
        self.assertIsInstance(pipeline.transforms[1], transforms.ToTensor)
        self.assertIsInstance(pipeline.transforms[2], MockPost)


class TestAddGaussianNoise(unittest.TestCase):
    """Tests for AddGaussianNoise."""

    def test_noise_adds_perturbation_and_clips(self) -> None:
        """Adding zero noise preserves tensor; noise clips to [0, 1]."""
        tensor = torch.tensor([[0.0, 1.0], [0.5, 0.2]])
        no_noise = AddGaussianNoise(mean=0.0, std=0.0)
        self.assertTrue(torch.allclose(no_noise(tensor), tensor))

        noisy = AddGaussianNoise(mean=10.0, std=1.0, clip=True)(tensor)
        self.assertTrue(torch.all(noisy <= 1.0))
        self.assertTrue(torch.all(noisy >= 0.0))

    def test_reproducibility_with_generator(self) -> None:
        """Using a seeded generator produces identical noise."""
        tensor = torch.zeros(1, 4, 4)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        noise1 = AddGaussianNoise(mean=0.0, std=0.2, generator=gen1)(tensor)
        noise2 = AddGaussianNoise(mean=0.0, std=0.2, generator=gen2)(tensor)
        self.assertTrue(torch.allclose(noise1, noise2))


class NoisyToyDataset(Dataset):
    """Mid-gray images passed through AddGaussianNoise, which uses the global RNG."""

    def __init__(self, size: int = TOY_DATASET_SIZE) -> None:
        """Builds size identical images, so any difference comes from the noise."""
        self.images = torch.full((size, 1, 4, 4), 0.5)
        self.labels = torch.arange(size)
        self.noise = AddGaussianNoise(mean=0.0, std=0.2)

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a freshly noised image and its label."""
        return self.noise(self.images[index]), self.labels[index]


class TestPerSampleSeededDataset(unittest.TestCase):
    """Tests for PerSampleSeededDataset."""

    def setUp(self) -> None:
        """Builds a dataset whose samples change on every unseeded read."""
        self.noisy = NoisyToyDataset()

    def test_unwrapped_random_transform_changes_on_every_read(self) -> None:
        """Without the wrapper, the same sample gets new noise each time."""
        self.assertFalse(torch.equal(self.noisy[0][0], self.noisy[0][0]))

    def test_preserves_length_and_labels(self) -> None:
        """Wrapping changes neither the size nor the labels."""
        seeded = PerSampleSeededDataset(self.noisy)
        self.assertEqual(len(seeded), len(self.noisy))
        for index in range(len(seeded)):
            self.assertEqual(seeded[index][1], self.noisy.labels[index])

    def test_same_sample_always_gets_the_same_output(self) -> None:
        """Reading a sample twice, or through another wrapper, gives equal images."""
        first = PerSampleSeededDataset(self.noisy, seed=7)
        second = PerSampleSeededDataset(self.noisy, seed=7)
        self.assertTrue(torch.equal(first[3][0], first[3][0]))
        self.assertTrue(torch.equal(first[3][0], second[3][0]))

    def test_different_samples_get_different_output(self) -> None:
        """Identical clean images end up with different noise."""
        seeded = PerSampleSeededDataset(self.noisy)
        self.assertFalse(torch.equal(seeded[0][0], seeded[1][0]))

    def test_different_seeds_give_different_output(self) -> None:
        """The base seed selects which fixed noise each sample gets."""
        first = PerSampleSeededDataset(self.noisy, seed=7)
        second = PerSampleSeededDataset(self.noisy, seed=8)
        self.assertFalse(torch.equal(first[0][0], second[0][0]))

    def test_does_not_touch_the_global_rng(self) -> None:
        """Reading samples leaves the global RNG state unchanged."""
        seeded = PerSampleSeededDataset(self.noisy)
        state = torch.get_rng_state()
        _ = [seeded[index] for index in range(len(seeded))]
        self.assertTrue(torch.equal(state, torch.get_rng_state()))



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


class TestCreateDataloaders(unittest.TestCase):
    """Tests for create_dataloaders."""

    def setUp(self) -> None:
        """Creates a training dataset and a smaller stand-in test dataset."""
        self.train_dataset = make_toy_dataset()
        self.test_dataset = make_toy_dataset(TOY_TEST_DATASET_SIZE)

    def test_without_val_split_there_is_no_validation_loader(self) -> None:
        """Leaving val_split at 0 keeps all training data and returns no val loader."""
        train_loader, val_loader, test_loader = create_dataloaders(self.train_dataset)

        self.assertIsNone(val_loader)
        self.assertIsNone(test_loader)
        self.assertEqual(len(train_loader.dataset), TOY_DATASET_SIZE)

    def test_with_val_split_gives_three_loaders_of_the_right_size(self) -> None:
        """A val_split carves the validation data out of the training data."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.train_dataset, test_dataset=self.test_dataset, val_split=0.2
        )

        self.assertEqual(len(train_loader.dataset), 8)
        self.assertEqual(len(val_loader.dataset), 2)
        self.assertEqual(len(test_loader.dataset), TOY_TEST_DATASET_SIZE)

    def test_test_dataset_alone_needs_no_val_split(self) -> None:
        """A test set can be loaded without holding out any validation data."""
        _, val_loader, test_loader = create_dataloaders(
            self.train_dataset, test_dataset=self.test_dataset
        )

        self.assertIsNone(val_loader)
        self.assertEqual(len(test_loader.dataset), TOY_TEST_DATASET_SIZE)

    def test_eval_batch_size_applies_only_to_evaluation_loaders(self) -> None:
        """Training keeps batch_size while validation and test use eval_batch_size."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.train_dataset,
            test_dataset=self.test_dataset,
            val_split=0.2,
            batch_size=4,
            eval_batch_size=2,
        )

        self.assertEqual(train_loader.batch_size, 4)
        self.assertEqual(val_loader.batch_size, 2)
        self.assertEqual(test_loader.batch_size, 2)

    def test_eval_batch_size_defaults_to_batch_size(self) -> None:
        """Omitting eval_batch_size reuses the training batch size."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.train_dataset,
            test_dataset=self.test_dataset,
            val_split=0.2,
            batch_size=4,
        )

        self.assertEqual(train_loader.batch_size, 4)
        self.assertEqual(val_loader.batch_size, 4)
        self.assertEqual(test_loader.batch_size, 4)

    def test_train_samples_randomly_and_evaluation_does_not(self) -> None:
        """Only the training loader draws samples in random order."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.train_dataset, test_dataset=self.test_dataset, val_split=0.2
        )

        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self.assertIsInstance(test_loader.sampler, SequentialSampler)

    def test_evaluation_loader_preserves_dataset_order(self) -> None:
        """Test labels come out in dataset order, so metrics are reproducible."""
        _, _, test_loader = create_dataloaders(
            self.train_dataset, test_dataset=self.test_dataset, batch_size=2
        )

        self.assertEqual(
            collect_labels(test_loader), list(range(TOY_TEST_DATASET_SIZE))
        )

    def test_training_loader_reorders_samples(self) -> None:
        """Training labels are the same set as the dataset but not in its order."""
        train_loader, _, _ = create_dataloaders(self.train_dataset, batch_size=2)

        labels = collect_labels(train_loader)

        self.assertEqual(sorted(labels), list(range(TOY_DATASET_SIZE)))
        self.assertNotEqual(labels, list(range(TOY_DATASET_SIZE)))

    def test_same_seed_gives_the_same_training_order(self) -> None:
        """Two loaders built with the same seed iterate in the same order."""
        first_loader, _, _ = create_dataloaders(
            self.train_dataset, batch_size=2, seed=7
        )
        second_loader, _, _ = create_dataloaders(
            self.train_dataset, batch_size=2, seed=7
        )

        self.assertEqual(collect_labels(first_loader), collect_labels(second_loader))

    def test_iterating_loaders_leaves_the_global_rng_untouched(self) -> None:
        """Loaders must not consume the RNG that dropout and augmentation use."""
        loaders = create_dataloaders(
            self.train_dataset, test_dataset=self.test_dataset, val_split=0.2
        )

        torch.manual_seed(0)
        expected = torch.rand(3)

        torch.manual_seed(0)
        for loader in loaders:
            for _ in loader:
                pass
        actual = torch.rand(3)

        self.assertTrue(torch.equal(expected, actual))

    def test_zero_batch_size_raises(self) -> None:
        """A batch size of 0 would yield empty batches forever."""
        with self.assertRaises(ValueError):
            create_dataloaders(self.train_dataset, batch_size=0)

    def test_zero_eval_batch_size_raises(self) -> None:
        """The evaluation batch size is validated just like the training one."""
        with self.assertRaises(ValueError):
            create_dataloaders(self.train_dataset, eval_batch_size=0)


if __name__ == "__main__":
    unittest.main()
