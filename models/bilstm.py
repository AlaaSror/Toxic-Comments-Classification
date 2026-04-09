"""
bilstm.py
---------
Custom Bidirectional LSTM with self-attention for multi-label
toxic comment classification. Built from scratch in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Additive self-attention over LSTM hidden states.
    Learns which tokens to focus on for the final representation.

    Args:
        hidden_dim : size of the LSTM hidden state (both directions)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output : (batch, seq_len, hidden_dim)
        Returns:
            context     : (batch, hidden_dim) — weighted sum of hidden states
        """
        scores  = self.attention(lstm_output).squeeze(-1)   # (batch, seq_len)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)   # (batch, seq_len, 1)
        context = (lstm_output * weights).sum(dim=1)        # (batch, hidden_dim)
        return context


class ToxicBiLSTM(nn.Module):
    """
    Custom BiLSTM + Attention classifier for multi-label toxicity detection.

    Architecture:
        Embedding → BiLSTM (2 layers) → Self-Attention → Dropout → Linear → Sigmoid

    Args:
        vocab_size    : number of tokens in vocabulary
        embed_dim     : embedding dimensionality
        hidden_dim    : LSTM hidden size (per direction)
        num_layers    : number of stacked LSTM layers
        num_labels    : number of output labels (7 for civil_comments)
        dropout       : dropout probability
        pad_idx       : index of the PAD token (for embedding masking)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_labels: int = 7,
        dropout:    float = 0.3,
        pad_idx:    int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # hidden_dim * 2 because bidirectional
        self.attention = SelfAttention(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len) — token indices

        Returns:
            logits : (batch, num_labels) — raw scores (apply sigmoid for probs)
        """
        embedded    = self.dropout(self.embedding(x))       # (batch, seq, embed)
        lstm_out, _ = self.lstm(embedded)                   # (batch, seq, hidden*2)
        context     = self.attention(lstm_out)              # (batch, hidden*2)
        context     = self.dropout(context)
        logits      = self.fc(context)                      # (batch, num_labels)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities (for inference)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
