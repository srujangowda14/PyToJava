import torch
import torch.nn as nn
from typing import Tuple, List
import torch.nn.functional as F

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
    
class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.
 
    score(h_t, h_s) = v^T · tanh(W_a·h_t + U_a·h_s)
 
    Returns:
        context   : weighted sum of encoder outputs  [B, 2H]
        attn_w    : attention weights                 [B, src_len]
    """
 
    def __init__(self, hidden_dim: int, enc_out_dim: int):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim,   hidden_dim, bias=False)
        self.U_a = nn.Linear(enc_out_dim,  hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim,   1,          bias=False)
 
    def forward(
        self,
        decoder_hidden:  torch.Tensor,   # [B, H]
        encoder_outputs: torch.Tensor,   # [B, S, 2H]
        src_mask:        torch.Tensor,   # [B, S]  True = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
 
        src_len = encoder_outputs.size(1)
 
        # Expand decoder hidden to match seq dim: [B, S, H]
        h_t = self.W_a(decoder_hidden).unsqueeze(1).repeat(1, src_len, 1)
        # Encoder side: [B, S, H]
        h_s = self.U_a(encoder_outputs)
 
        energy = self.v(torch.tanh(h_t + h_s)).squeeze(2)   # [B, S]
 
        # Mask padding positions with -inf before softmax
        energy = energy.masked_fill(src_mask, float("-inf"))
 
        attn_weights = F.softmax(energy, dim=1)              # [B, S]
 
        # Weighted sum: [B, 1, S] × [B, S, 2H] → [B, 1, 2H]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)                         # [B, 2H]
 
        return context, attn_weights
    
class Decoder(nn.Module):
    """
    Unidirectional GRU decoder with input-feeding attention.
 
    At each step:
      1. Embed the previous token
      2. Concatenate with previous context (input feeding)
      3. Feed through GRU
      4. Compute attention over encoder outputs
      5. Project [hidden ⊕ context] → vocab logits
    """
 
    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int,
        hidden_dim:   int,
        enc_out_dim:  int,   # 2*hidden_dim (bidirectional encoder)
        n_layers:     int,
        dropout:      float = 0.3,
        pad_idx:      int   = 0,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.vocab_size  = vocab_size
        self.n_layers    = n_layers
 
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout   = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim, enc_out_dim)
 
        # Input = embed + previous context
        self.rnn = nn.GRU(
            input_size  = embed_dim + enc_out_dim,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0,
        )
 
        # Projection: [hidden ⊕ context] → vocab
        self.fc_out = nn.Linear(hidden_dim + enc_out_dim, vocab_size)
 
    def forward_step(
        self,
        token:           torch.Tensor,   # [B]       current token ids
        hidden:          torch.Tensor,   # [n_layers, B, H]
        encoder_outputs: torch.Tensor,   # [B, S, 2H]
        src_mask:        torch.Tensor,   # [B, S]
        prev_context:    torch.Tensor,   # [B, 2H]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits   : [B, vocab_size]
            hidden   : [n_layers, B, H]
            attn_w   : [B, src_len]
        """
        # top hidden layer for attention query
        query = hidden[-1]                              # [B, H]
 
        # Attention
        context, attn_w = self.attention(query, encoder_outputs, src_mask)
 
        # Embed + concat context (input feeding)
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))  # [B, 1, E]
        rnn_input = torch.cat(
            [embedded, context.unsqueeze(1)], dim=2
        )                                               # [B, 1, E+2H]
 
        output, hidden = self.rnn(rnn_input, hidden)   # output: [B, 1, H]
        output = output.squeeze(1)                     # [B, H]
 
        # Project
        logits = self.fc_out(
            torch.cat([output, context], dim=1)
        )                                               # [B, vocab_size]
 
        return logits, hidden, attn_w
    
class Seq2SeqTranslator(nn.Module):
    """
    End-to-end Python → Java seq2seq model.
 
    Training  : teacher forcing with configurable ratio
    Inference : greedy or beam search decoding
    """
 
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim:      int   = 256,
        hidden_dim:     int   = 512,
        n_layers:       int   = 2,
        dropout:        float = 0.3,
        src_pad_idx:    int   = 0,
        tgt_pad_idx:    int   = 0,
        sos_idx:        int   = 2,
        eos_idx:        int   = 3,
    ):
        super().__init__()
 
        enc_out_dim = hidden_dim * 2   # bidirectional
 
        self.encoder = Encoder(
            vocab_size  = src_vocab_size,
            embed_dim   = embed_dim,
            hidden_dim  = hidden_dim,
            n_layers    = n_layers,
            dropout     = dropout,
            pad_idx     = src_pad_idx,
        )
 
        self.decoder = Decoder(
            vocab_size   = tgt_vocab_size,
            embed_dim    = embed_dim,
            hidden_dim   = hidden_dim,
            enc_out_dim  = enc_out_dim,
            n_layers     = n_layers,
            dropout      = dropout,
            pad_idx      = tgt_pad_idx,
        )
 
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.sos_idx     = sos_idx
        self.eos_idx     = eos_idx
 
        self._init_weights()
 
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
 
    # ── Training forward pass ─────────────────────────────────────────────────
 
    def forward(
        self,
        src:              torch.Tensor,   # [B, src_len]
        tgt:              torch.Tensor,   # [B, tgt_len]
        src_mask:         torch.Tensor,   # [B, src_len]
        teacher_force_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Returns logits: [B, tgt_len-1, tgt_vocab_size]
        Target for loss: tgt[:, 1:]  (shifted by 1, excluding <SOS>)
        """
        batch_size  = src.size(0)
        tgt_len     = tgt.size(1)
        tgt_vocab   = self.decoder.vocab_size
 
        # Encode
        enc_outputs, hidden = self.encoder(src, src_mask)
 
        # Initial decoder token = <SOS>
        dec_input   = tgt[:, 0]                            # [B]
        prev_context = torch.zeros(
            batch_size, self.encoder.hidden_dim * 2,
            device=src.device
        )
 
        all_logits = []
 
        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder.forward_step(
                dec_input, hidden, enc_outputs, src_mask, prev_context
            )
            all_logits.append(logits.unsqueeze(1))         # [B, 1, V]
 
            # Teacher forcing
            if torch.rand(1).item() < teacher_force_ratio:
                dec_input = tgt[:, t]
            else:
                dec_input = logits.argmax(dim=1)
 
        return torch.cat(all_logits, dim=1)                # [B, tgt_len-1, V]
 
    # ── Greedy decoding ───────────────────────────────────────────────────────
 
    @torch.no_grad()
    def translate_greedy(
        self,
        src:     torch.Tensor,   # [1, src_len]
        src_mask: torch.Tensor,  # [1, src_len]
        max_len: int = 768,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Greedy decoding for a single example.
        Returns:
            token_ids   : list of generated token indices (excl. SOS/EOS)
            attn_matrix : [steps, src_len] attention weights
        """
        self.eval()
        enc_outputs, hidden = self.encoder(src, src_mask)
 
        dec_input    = torch.tensor([self.sos_idx], device=src.device)
        prev_context = torch.zeros(1, self.encoder.hidden_dim * 2, device=src.device)
 
        generated    = []
        attn_matrix  = []
 
        for _ in range(max_len):
            logits, hidden, attn_w = self.decoder.forward_step(
                dec_input, hidden, enc_outputs, src_mask, prev_context
            )
            pred = logits.argmax(dim=1)   # [1]
            token_id = pred.item()
 
            if token_id == self.eos_idx:
                break
 
            generated.append(token_id)
            attn_matrix.append(attn_w.squeeze(0).cpu())
            dec_input = pred
 
        attn_tensor = torch.stack(attn_matrix) if attn_matrix else torch.zeros(1, src.size(1))
        return generated, attn_tensor
 
    # ── Beam search decoding ──────────────────────────────────────────────────
 
    @torch.no_grad()
    def translate_beam(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
        beam_size: int = 4,
        max_len:   int = 768,
    ) -> List[int]:
        """
        Beam search decoding. Returns best hypothesis token ids.
        """
        self.eval()
        enc_outputs, hidden = self.encoder(src, src_mask)
        prev_context = torch.zeros(1, self.encoder.hidden_dim * 2, device=src.device)
 
        # Each beam: (log_prob, token_ids, hidden, prev_context)
        beams = [(0.0, [self.sos_idx], hidden, prev_context)]
        completed = []
 
        for _ in range(max_len):
            new_beams = []
            for log_prob, tokens, h, ctx in beams:
                if tokens[-1] == self.eos_idx:
                    completed.append((log_prob, tokens[1:-1]))
                    continue
 
                dec_input = torch.tensor([tokens[-1]], device=src.device)
                logits, new_h, _ = self.decoder.forward_step(
                    dec_input, h, enc_outputs, src_mask, ctx
                )
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)  # [V]
                top_log_probs, top_ids = log_probs.topk(beam_size)
 
                for lp, tid in zip(top_log_probs.tolist(), top_ids.tolist()):
                    new_beams.append((
                        log_prob + lp,
                        tokens + [tid],
                        new_h,
                        ctx,
                    ))
 
            if not new_beams:
                break
 
            # Keep top beam_size beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
 
        # Add any unfinished beams
        for log_prob, tokens, _, _ in beams:
            completed.append((log_prob, tokens[1:]))
 
        if not completed:
            return []
 
        best = max(completed, key=lambda x: x[0])
        return best[1]
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
