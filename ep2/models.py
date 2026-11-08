"""Character-level language models compared in this assignment.

CharMLP sees the last `context` characters and predicts the next one:
x: (B, context) -> logits: (B, V).

CharRNN reads a whole sequence and predicts the next character at every step:
x: (B, T) -> logits: (B, T, V).
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

CELL_TYPES = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
GATES_PER_CELL = {"rnn": 1, "gru": 3, "lstm": 4}

HiddenState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Samples one token id per row from (B, V) logits.

    Temperatures below 1 sharpen the distribution (safer, more repetitive
    text) and above 1 flatten it (more diverse, more errors).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _check_prompt(prompt: torch.Tensor, min_length: int) -> None:
    """Raises if the prompt is not a 1-D tensor of at least min_length tokens."""
    if prompt.dim() != 1:
        raise ValueError(f"prompt must be 1-D, got shape {tuple(prompt.shape)}")
    if len(prompt) < min_length:
        raise ValueError(
            f"prompt must have at least {min_length} tokens, got {len(prompt)}"
        )


class CharMLP(nn.Module):
    """Feed-forward model on a fixed window"""

    def __init__(
        self,
        vocab_size: int,
        context: int,
        embed_dim: int = 24,
        hidden_size: int = 256,
    ) -> None:
        """Builds the embedding and the two fully connected layers."""
        super().__init__()
        self.vocab_size = vocab_size
        self.context = context
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context * embed_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps (B, context) windows to (B, V) logits."""
        if x.dim() != 2 or x.shape[1] != self.context:
            raise ValueError(
                f"expected (B, {self.context}) windows, got shape {tuple(x.shape)}"
            )
        return self.layers(self.embedding(x))

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Extends a prompt of at least `context` tokens by sliding the window."""
        _check_prompt(prompt, self.context)
        was_training = self.training
        self.eval()
        tokens = prompt.tolist()
        for _ in range(max_new_tokens):
            window = torch.tensor([tokens[-self.context :]], dtype=torch.long)
            next_id = sample_next(self(window), temperature, generator)
            tokens.append(next_id.item())
        self.train(was_training)
        return torch.tensor(tokens, dtype=torch.long)


class CharRNN(nn.Module):
    """Recurrent model"""

    def __init__(
        self,
        vocab_size: int,
        cell: str = "lstm",
        embed_dim: int = 24,
        hidden_size: int = 256,
        num_layers: int = 1,
    ) -> None:
        """Builds the embedding, the recurrent stack and the output layer."""
        super().__init__()
        if cell not in CELL_TYPES:
            raise ValueError(f"cell must be one of {sorted(CELL_TYPES)}, got {cell!r}")
        self.vocab_size = vocab_size
        self.cell = cell
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = CELL_TYPES[cell](
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward_with_state(
        self, x: torch.Tensor, hidden: Optional[HiddenState] = None
    ) -> Tuple[torch.Tensor, HiddenState]:
        """Maps (B, T) tokens to (B, T, V) logits and the final hidden state."""
        if x.dim() != 2:
            raise ValueError(f"expected (B, T) tokens, got shape {tuple(x.shape)}")
        outputs, hidden = self.rnn(self.embedding(x), hidden)
        return self.head(outputs), hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps (B, T) tokens to (B, T, V) logits, starting from a zero state."""
        logits, _ = self.forward_with_state(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Extends a non-empty prompt, carrying the hidden state across steps."""
        _check_prompt(prompt, 1)
        was_training = self.training
        self.eval()
        logits, hidden = self.forward_with_state(prompt.unsqueeze(0))
        tokens = prompt.tolist()
        for _ in range(max_new_tokens):
            next_id = sample_next(logits[:, -1], temperature, generator)
            tokens.append(next_id.item())
            logits, hidden = self.forward_with_state(next_id.view(1, 1), hidden)
        self.train(was_training)
        return torch.tensor(tokens, dtype=torch.long)
