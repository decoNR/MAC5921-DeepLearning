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
    NextCharDataset,
    SequenceDataset,
    build_corpus,
    build_datasets,
    create_text_dataloaders,
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


class TestNextCharDataset(unittest.TestCase):
    """Tests for the window -> next character dataset."""

    def setUp(self) -> None:
        """Uses 0..9 as tokens so that targets are easy to reason about."""
        self.data = torch.arange(10)

    def test_length(self) -> None:
        """Every token after the first `context` ones is a target once."""
        self.assertEqual(len(NextCharDataset(self.data, 3)), 7)

    def test_items(self) -> None:
        """The target is the token right after the window."""
        dataset = NextCharDataset(self.data, 3)
        x, y = dataset[0]
        self.assertTrue(torch.equal(x, torch.tensor([0, 1, 2])))
        self.assertEqual(y.item(), 3)
        x, y = dataset[-1]
        self.assertTrue(torch.equal(x, torch.tensor([6, 7, 8])))
        self.assertEqual(y.item(), 9)

    def test_shapes_and_dtype(self) -> None:
        """x has shape (context,), y is a scalar, both long."""
        x, y = NextCharDataset(self.data, 4)[2]
        self.assertEqual(x.shape, (4,))
        self.assertEqual(y.shape, ())
        self.assertEqual(x.dtype, torch.long)

    def test_context_one_is_bigram(self) -> None:
        """With context=1 every pair of consecutive tokens is an example."""
        dataset = NextCharDataset(self.data, 1)
        self.assertEqual(len(dataset), 9)
        pairs = [(x.item(), y.item()) for x, y in dataset]
        self.assertEqual(pairs, [(i, i + 1) for i in range(9)])

    def test_out_of_range_raises(self) -> None:
        """Indexing past the end raises IndexError."""
        dataset = NextCharDataset(self.data, 3)
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_invalid_arguments_raise(self) -> None:
        """Non-positive contexts and too-short data are rejected."""
        with self.assertRaises(ValueError):
            NextCharDataset(self.data, 0)
        with self.assertRaises(ValueError):
            NextCharDataset(self.data, 10)
        with self.assertRaises(TypeError):
            NextCharDataset(self.data, 2.0)


class TestSequenceDataset(unittest.TestCase):
    """Tests for the sequence -> shifted sequence dataset."""

    def setUp(self) -> None:
        """Uses 0..10 as tokens so that targets are easy to reason about."""
        self.data = torch.arange(11)

    def test_target_is_input_shifted_by_one(self) -> None:
        """y[t] is the token after x[t]."""
        x, y = SequenceDataset(self.data, 4)[0]
        self.assertTrue(torch.equal(x, torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.equal(y, torch.tensor([1, 2, 3, 4])))

    def test_default_stride_gives_non_overlapping_chunks(self) -> None:
        """11 tokens with seq_len=4 yield chunks starting at 0 and 4."""
        dataset = SequenceDataset(self.data, 4)
        self.assertEqual(len(dataset), 2)
        x, y = dataset[1]
        self.assertTrue(torch.equal(x, torch.tensor([4, 5, 6, 7])))
        self.assertTrue(torch.equal(y, torch.tensor([5, 6, 7, 8])))

    def test_last_chunk_fits_exactly(self) -> None:
        """With 9 tokens and seq_len=4, the second target ends at the last token."""
        dataset = SequenceDataset(torch.arange(9), 4)
        self.assertEqual(len(dataset), 2)
        _, y = dataset[-1]
        self.assertEqual(y[-1].item(), 8)

    def test_custom_stride(self) -> None:
        """A stride of 1 yields every possible (overlapping) chunk."""
        dataset = SequenceDataset(self.data, 4, stride=1)
        self.assertEqual(len(dataset), 7)
        x, _ = dataset[6]
        self.assertTrue(torch.equal(x, torch.tensor([6, 7, 8, 9])))

    def test_out_of_range_raises(self) -> None:
        """Indexing past the end raises IndexError."""
        dataset = SequenceDataset(self.data, 4)
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_invalid_arguments_raise(self) -> None:
        """Non-positive lengths/strides and too-short data are rejected."""
        with self.assertRaises(ValueError):
            SequenceDataset(self.data, 0)
        with self.assertRaises(ValueError):
            SequenceDataset(self.data, 4, stride=0)
        with self.assertRaises(ValueError):
            SequenceDataset(self.data, 11)


class TestBuildDatasets(unittest.TestCase):
    """Tests for building datasets for every split."""

    def setUp(self) -> None:
        """Builds a corpus from a repeated toy text, long enough for every split."""
        self.corpus = build_corpus(TOY_TEXT * 10, 0.8, 0.1)

    def test_next_char_mode(self) -> None:
        """next_char mode wraps each split in a NextCharDataset."""
        datasets = build_datasets(self.corpus, "next_char", 5)
        splits = (self.corpus.train_data, self.corpus.val_data, self.corpus.test_data)
        for dataset, split in zip(datasets, splits):
            self.assertIsInstance(dataset, NextCharDataset)
            self.assertEqual(len(dataset), len(split) - 5)

    def test_sequence_mode(self) -> None:
        """sequence mode wraps each split in a SequenceDataset."""
        for dataset in build_datasets(self.corpus, "sequence", 8):
            self.assertIsInstance(dataset, SequenceDataset)
            self.assertEqual(dataset.seq_len, 8)

    def test_unknown_mode_raises(self) -> None:
        """Unknown modes are rejected."""
        with self.assertRaises(ValueError):
            build_datasets(self.corpus, "images", 5)


class TestCreateTextDataloaders(unittest.TestCase):
    """Tests for the text dataloaders."""

    def setUp(self) -> None:
        """Builds small sequence datasets from 0..99."""
        data = torch.arange(100)
        self.train = SequenceDataset(data[:80], 4)
        self.val = SequenceDataset(data[80:90], 4)
        self.test = SequenceDataset(data[90:], 4)

    def test_batch_shapes(self) -> None:
        """Batches stack sequences into (batch, seq_len)."""
        train_loader, _, _ = create_text_dataloaders(
            self.train, self.val, self.test, batch_size=5
        )
        x, y = next(iter(train_loader))
        self.assertEqual(x.shape, (5, 4))
        self.assertEqual(y.shape, (5, 4))

    def test_eval_loaders_are_not_shuffled(self) -> None:
        """Validation batches keep the original order."""
        _, val_loader, _ = create_text_dataloaders(
            self.train, self.val, self.test, batch_size=5
        )
        x, _ = next(iter(val_loader))
        self.assertTrue(torch.equal(x[0], torch.tensor([80, 81, 82, 83])))

    def test_train_shuffle_is_reproducible(self) -> None:
        """The same seed yields the same training order."""
        orders = []
        for _ in range(2):
            loader, _, _ = create_text_dataloaders(
                self.train, self.val, self.test, batch_size=20, seed=7
            )
            orders.append(next(iter(loader))[0])
        self.assertTrue(torch.equal(orders[0], orders[1]))

    def test_invalid_batch_size_raises(self) -> None:
        """Non-positive batch sizes are rejected."""
        with self.assertRaises(ValueError):
            create_text_dataloaders(self.train, self.val, self.test, batch_size=0)


if __name__ == "__main__":
    unittest.main()
