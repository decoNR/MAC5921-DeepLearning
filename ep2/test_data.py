"""Tests for the Tiny Shakespeare data utilities."""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ep2.data import (
    FILE_NAME,
    CharVocabulary,
    build_corpus,
    download_tiny_shakespeare,
    load_text,
    split_data,
)

# A short excerpt with letters, punctuation and newlines, as in the real corpus.
TOY_TEXT = "First Citizen:\nBefore we proceed, hear me speak.\n"


class TestCharVocabulary(unittest.TestCase):
    """Tests for the character vocabulary."""

    def setUp(self) -> None:
        """Builds a vocabulary from the toy text."""
        self.vocab = CharVocabulary.from_text(TOY_TEXT)

    def test_size_matches_unique_characters(self) -> None:
        """Vocabulary size equals the number of distinct characters."""
        self.assertEqual(self.vocab.size, len(set(TOY_TEXT)))
        self.assertEqual(len(self.vocab), self.vocab.size)

    def test_characters_are_sorted(self) -> None:
        """Token ids follow the sorted character order, making them deterministic."""
        self.assertEqual(self.vocab.itos, sorted(set(TOY_TEXT)))

    def test_stoi_and_itos_are_inverse(self) -> None:
        """stoi and itos are inverse mappings."""
        for i, ch in enumerate(self.vocab.itos):
            self.assertEqual(self.vocab.stoi[ch], i)

    def test_encode_returns_long_tensor(self) -> None:
        """Encoding yields a 1-D long tensor with one id per character."""
        ids = self.vocab.encode(TOY_TEXT)
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(ids.shape, (len(TOY_TEXT),))

    def test_decode_inverts_encode(self) -> None:
        """Decoding an encoded string recovers the original string."""
        self.assertEqual(self.vocab.decode(self.vocab.encode(TOY_TEXT)), TOY_TEXT)

    def test_decode_accepts_list(self) -> None:
        """Decoding also works with a plain list of ids."""
        ids = self.vocab.encode("hear").tolist()
        self.assertEqual(self.vocab.decode(ids), "hear")

    def test_encode_unknown_character_raises(self) -> None:
        """Encoding a character outside the vocabulary raises KeyError."""
        with self.assertRaises(KeyError):
            self.vocab.encode("Q")

    def test_duplicate_characters_rejected(self) -> None:
        """A vocabulary with repeated characters is invalid."""
        with self.assertRaises(ValueError):
            CharVocabulary(["a", "a"])

    def test_multi_character_entries_rejected(self) -> None:
        """Every vocabulary entry must be exactly one character."""
        with self.assertRaises(ValueError):
            CharVocabulary(["ab"])


class TestSplitData(unittest.TestCase):
    """Tests for the contiguous train/validation/test split."""

    def test_split_sizes(self) -> None:
        """An 80/10/10 split of 100 tokens yields 80, 10 and 10 tokens."""
        train, val, test = split_data(torch.arange(100), 0.8, 0.1)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(test), 10)

    def test_split_is_contiguous_and_complete(self) -> None:
        """Concatenating the three parts recovers the original order."""
        data = torch.arange(100)
        train, val, test = split_data(data, 0.8, 0.1)
        self.assertTrue(torch.equal(torch.cat([train, val, test]), data))

    def test_test_split_receives_remainder(self) -> None:
        """The test split receives whatever train and validation leave."""
        _, _, test = split_data(torch.arange(100), 0.7, 0.1)
        self.assertEqual(len(test), 20)

    def test_invalid_fractions_raise(self) -> None:
        """Non-positive fractions, or fractions leaving no test split, are rejected."""
        cases = ((0.0, 0.1), (0.8, 0.0), (-0.1, 0.1), (0.9, 0.1), (1.0, 0.5))
        for train_fraction, val_fraction in cases:
            with self.subTest(train=train_fraction, val=val_fraction):
                with self.assertRaises(ValueError):
                    split_data(torch.arange(10), train_fraction, val_fraction)

    def test_empty_split_raises(self) -> None:
        """Data too short for all three splits to be non-empty is rejected."""
        with self.assertRaises(ValueError):
            split_data(torch.arange(3), 0.8, 0.1)


class TestBuildCorpus(unittest.TestCase):
    """Tests for corpus construction."""

    def test_corpus_round_trip(self) -> None:
        """Decoding the three splits back to back recovers the full text."""
        corpus = build_corpus(TOY_TEXT, 0.8, 0.1)
        decoded = "".join(
            corpus.vocab.decode(part)
            for part in (corpus.train_data, corpus.val_data, corpus.test_data)
        )
        self.assertEqual(decoded, TOY_TEXT)
        self.assertEqual(len(corpus.train_data), int(len(TOY_TEXT) * 0.8))


class TestDownload(unittest.TestCase):
    """Tests for the download helper, using a local file:// URL (no network)."""

    def setUp(self) -> None:
        """Creates a temporary source file and destination directory."""
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.txt"
        self.source.write_text(TOY_TEXT, encoding="utf-8")
        self.url = self.source.as_uri()

    def tearDown(self) -> None:
        """Removes the temporary directory."""
        self.tmp.cleanup()

    def test_download_writes_file(self) -> None:
        """Downloading copies the content into data_dir/input.txt."""
        path = download_tiny_shakespeare(self.root / "out", self.url)
        self.assertEqual(path, self.root / "out" / FILE_NAME)
        self.assertEqual(load_text(path), TOY_TEXT)
        self.assertFalse(path.with_suffix(".txt.part").exists())

    def test_download_skips_existing_file(self) -> None:
        """An existing file is reused instead of downloaded again."""
        out = self.root / "out"
        out.mkdir()
        (out / FILE_NAME).write_text("cached", encoding="utf-8")
        path = download_tiny_shakespeare(out, self.url)
        self.assertEqual(load_text(path), "cached")


if __name__ == "__main__":
    unittest.main()
