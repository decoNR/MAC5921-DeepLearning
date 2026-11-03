"""Character-level language models compared in this assignment.

CharMLP sees the last `context` characters and predicts the next one:
x: (B, context) -> logits: (B, V).
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn


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
        dropout: float = 0.0,
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
            nn.Dropout(dropout),
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
