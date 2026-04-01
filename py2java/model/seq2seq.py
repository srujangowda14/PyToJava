import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    """
    Bidirectional GRU encoder.
 
    Processes the Python token sequence and returns:
        outputs  : all hidden states  [batch, src_len, 2*hidden_dim]
        hidden   : final hidden state [n_layers, batch, hidden_dim]  (merged)
    """
 
    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int,
        hidden_dim:   int,
        n_layers:     int,
        dropout:      float = 0.3,
        pad_idx:      int   = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
 
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout   = nn.Dropout(dropout)
 
        self.rnn = nn.GRU(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if n_layers > 1 else 0,
        )
 
        # Project bidirectional hidden → unidirectional for decoder init
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
 
    def forward(
        self,
        src:      torch.Tensor,   # [batch, src_len]
        src_mask: torch.Tensor,   # [batch, src_len]  True = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
 
        embedded = self.dropout(self.embedding(src))   # [B, S, E]
        outputs, hidden = self.rnn(embedded)            # outputs: [B, S, 2H]
 
        # hidden shape: [2*n_layers, B, H]
        # Merge forward + backward for each layer → [n_layers, B, H]
        hidden = self._merge_hidden(hidden)
 
        return outputs, hidden
 
    def _merge_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Concatenate forward and backward final hidden states
        then project down to hidden_dim.
        hidden: [2*n_layers, B, H]
        """
        # Stack pairs: [n_layers, B, 2H]
        fwd = hidden[0::2]   # even indices = forward layers
        bwd = hidden[1::2]   # odd  indices = backward layers
        merged = torch.cat([fwd, bwd], dim=2)           # [n_layers, B, 2H]
        # Apply projection per layer
        projected = torch.tanh(self.fc(merged))         # [n_layers, B, H]
        return projected
    
