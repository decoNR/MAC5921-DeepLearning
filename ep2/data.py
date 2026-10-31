"""Tiny Shakespeare data utilities: download, vocabulary, split and datasets."""

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from library.dataset import DEFAULT_SEED

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_DIR = PROJECT_ROOT / "data" / "tiny_shakespeare"
FILE_NAME = "input.txt"
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_VAL_FRACTION = 0.1
DATASET_MODES = ("next_char", "sequence")


def download_tiny_shakespeare(
    data_dir: Union[str, Path] = DATA_DIR,
    url: str = TINY_SHAKESPEARE_URL,
) -> Path:
    """Downloads the Tiny Shakespeare text file, skipping it if already present."""
    data_dir = Path(data_dir)
    path = data_dir / FILE_NAME
    if path.exists():
        return path

    data_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response:
            partial.write_bytes(response.read())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def load_text(path: Union[str, Path]) -> str:
    """Reads a UTF-8 text file."""
    return Path(path).read_text(encoding="utf-8")


class CharVocabulary:
    """Bidirectional mapping between characters and integer token ids."""

    def __init__(self, chars: Iterable[str]) -> None:
        """Builds the vocabulary from an iterable of unique single characters."""
        self.itos: List[str] = list(chars)
        if any(len(ch) != 1 for ch in self.itos):
            raise ValueError("Every vocabulary entry must be a single character")
        if len(set(self.itos)) != len(self.itos):
            raise ValueError("Vocabulary characters must be unique")
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharVocabulary":
        """Builds a vocabulary from the sorted set of characters in the text."""
        return cls(sorted(set(text)))

    @property
    def size(self) -> int:
        """Returns the number of distinct characters."""
        return len(self.itos)

    def __len__(self) -> int:
        """Returns the number of distinct characters."""
        return self.size

    def encode(self, text: str) -> torch.Tensor:
        """Converts a string into a 1-D tensor of token ids."""
        try:
            ids = [self.stoi[ch] for ch in text]
        except KeyError as err:
            raise KeyError(f"Character {err.args[0]!r} is not in the vocabulary")
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Union[torch.Tensor, Iterable[int]]) -> str:
        """Converts a sequence of token ids back into a string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


def split_data(
    data: torch.Tensor,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a token sequence into contiguous train, validation and test parts"""
    if train_fraction <= 0.0 or val_fraction <= 0.0:
        raise ValueError(
            "train_fraction and val_fraction must be positive, got "
            f"{train_fraction} and {val_fraction}"
        )
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            "train_fraction + val_fraction must be < 1 to leave a test split, got "
            f"{train_fraction + val_fraction}"
        )
    train_end = int(len(data) * train_fraction)
    val_end = int(len(data) * (train_fraction + val_fraction))
    if train_end == 0 or val_end == train_end or val_end == len(data):
        raise ValueError("Data is too short to produce non-empty splits")
    return data[:train_end], data[train_end:val_end], data[val_end:]


@dataclass
class TextCorpus:
    """Encoded text corpus with its vocabulary and train/validation/test splits."""

    text: str
    vocab: CharVocabulary
    train_data: torch.Tensor
    val_data: torch.Tensor
    test_data: torch.Tensor


def build_corpus(
    text: str,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> TextCorpus:
    """Encodes a text and splits it into train, validation and test sequences."""
    vocab = CharVocabulary.from_text(text)
    train_data, val_data, test_data = split_data(
        vocab.encode(text), train_fraction, val_fraction
    )
    return TextCorpus(text, vocab, train_data, val_data, test_data)


def load_tiny_shakespeare(
    data_dir: Union[str, Path] = DATA_DIR,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> TextCorpus:
    """Downloads (if needed), encodes and splits the Tiny Shakespeare corpus."""
    path = download_tiny_shakespeare(data_dir)
    return build_corpus(load_text(path), train_fraction, val_fraction)


def _check_positive_int(name: str, value: int) -> None:
    """Raises if value is not a positive int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _normalize_index(idx: int, length: int) -> int:
    """Resolves negative indices and raises IndexError when out of range."""
    if idx < 0:
        idx += length
    if not 0 <= idx < length:
        raise IndexError(f"Index out of range for dataset of size {length}")
    return idx


class NextCharDataset(Dataset):
    """Window of `context` characters -> the single next character.

    Used by the bigram (context=1) and MLP models. Every position after the
    first `context` characters is a target exactly once.

    Example (context=3) on "hello": ("hel" -> "l"), ("ell" -> "o").
    """

    def __init__(self, data: torch.Tensor, context: int) -> None:
        """Wraps a 1-D token tensor."""
        _check_positive_int("context", context)
        if data.dim() != 1:
            raise ValueError(f"data must be 1-D, got shape {tuple(data.shape)}")
        if len(data) <= context:
            raise ValueError(
                f"data must be longer than context ({context}), got {len(data)}"
            )
        self.data = data
        self.context = context

    def __len__(self) -> int:
        """Returns the number of (window, next character) pairs."""
        return len(self.data) - self.context

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns x of shape (context,) and a scalar target y."""
        idx = _normalize_index(idx, len(self))
        x = self.data[idx : idx + self.context]
        y = self.data[idx + self.context]
        return x, y


class SequenceDataset(Dataset):
    """Sequence of `seq_len` characters -> the same sequence shifted by one.

    Used by the recurrent models, which predict the next character at every
    time step. With the default stride (= seq_len) the chunks do not overlap,
    so, as in NextCharDataset, each character is a target once per epoch.

    Example (seq_len=4) on "hello": ("hell" -> "ello").
    """

    def __init__(
        self, data: torch.Tensor, seq_len: int, stride: Optional[int] = None
    ) -> None:
        """Wraps a 1-D token tensor. A trailing remainder shorter than seq_len is dropped."""
        if stride is None:
            stride = seq_len
        _check_positive_int("seq_len", seq_len)
        _check_positive_int("stride", stride)
        if data.dim() != 1:
            raise ValueError(f"data must be 1-D, got shape {tuple(data.shape)}")
        if len(data) <= seq_len:
            raise ValueError(
                f"data must be longer than seq_len ({seq_len}), got {len(data)}"
            )
        self.data = data
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self) -> int:
        """Returns the number of (input, shifted target) sequence pairs."""
        # The target needs one extra token after the input, hence the - 1.
        return (len(self.data) - 1 - self.seq_len) // self.stride + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns x and y, both of shape (seq_len,), with y[t] == x[t + 1]."""
        idx = _normalize_index(idx, len(self))
        start = idx * self.stride
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def build_datasets(
    corpus: TextCorpus, mode: str, length: int
) -> Tuple[Dataset, Dataset, Dataset]:
    """Builds train, validation and test datasets for a model family.

    Args:
        corpus: Encoded and split corpus.
        mode: "next_char" (bigram/MLP) or "sequence" (RNN/LSTM/GRU).
        length: The context size for "next_char" or seq_len for "sequence".
    """
    if mode == "next_char":
        factory = NextCharDataset
    elif mode == "sequence":
        factory = SequenceDataset
    else:
        raise ValueError(f"mode must be one of {DATASET_MODES}, got {mode!r}")
    return tuple(
        factory(split, length)
        for split in (corpus.train_data, corpus.val_data, corpus.test_data)
    )


def create_text_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    eval_batch_size: Optional[int] = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wraps the three datasets into loaders; only the training one is shuffled."""
    if eval_batch_size is None:
        eval_batch_size = batch_size
    _check_positive_int("batch_size", batch_size)
    _check_positive_int("eval_batch_size", eval_batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    corpus = load_tiny_shakespeare()
    print(f"Characters: {len(corpus.text):,}")
    print(f"Vocabulary size: {corpus.vocab.size}")
    print(f"Vocabulary: {''.join(corpus.vocab.itos)!r}")
    print(f"Train tokens: {len(corpus.train_data):,}")
    print(f"Validation tokens: {len(corpus.val_data):,}")
    print(f"Test tokens: {len(corpus.test_data):,}")

    for mode, length in (("next_char", 8), ("sequence", 100)):
        train_ds, val_ds, test_ds = build_datasets(corpus, mode, length)
        x, y = train_ds[0]
        print(
            f"\n{mode} (length={length}): "
            f"{len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,} examples"
        )
        print(f"  x: {corpus.vocab.decode(x)!r}")
        print(f"  y: {corpus.vocab.decode(y.reshape(-1))!r}")

