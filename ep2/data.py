"""Tiny Shakespeare data utilities: download, character vocabulary and split."""

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_DIR = PROJECT_ROOT / "data" / "tiny_shakespeare"
FILE_NAME = "input.txt"
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_VAL_FRACTION = 0.1


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


if __name__ == "__main__":
    corpus = load_tiny_shakespeare()
    print(f"Characters: {len(corpus.text):,}")
    print(f"Vocabulary size: {corpus.vocab.size}")
    print(f"Vocabulary: {''.join(corpus.vocab.itos)!r}")
    print(f"Train tokens: {len(corpus.train_data):,}")
    print(f"Validation tokens: {len(corpus.val_data):,}")
    print(f"Test tokens: {len(corpus.test_data):,}")

