"""Tests for the generic dataset pipeline utilities."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from torchvision import transforms

from ep1.dataset import get_pipeline

# A 2x2 grayscale image with known pixel values, used as a didactic example.
TOY_PIXELS = [0, 255, 0, 255]
TOY_MEAN = (0.5,)
TOY_STD = (0.5,)


def make_toy_image() -> Image.Image:
    """Builds a tiny 2x2 grayscale image with pixels 0 and 255."""
    image = Image.new("L", (2, 2))
    image.putdata(TOY_PIXELS)
    return image


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


if __name__ == "__main__":
    unittest.main()
