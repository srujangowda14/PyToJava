import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    Improves generalisation by preventing the model from being overconfident.
 
    smoothing=0.1 is the standard choice (used in the original Transformer paper).
    """
 
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B * T, V]
        targets : [B * T]
        """
        V = self.vocab_size
        # Smooth target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(logits, self.smoothing / (V - 2))
            smooth_dist[:, self.pad_idx] = 0
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            mask = (targets == self.pad_idx)
            smooth_dist[mask] = 0
 
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(smooth_dist * log_probs).sum(dim=1)
 
        # Average over non-padding tokens
        non_pad = (~mask).sum()
        return loss.sum() / non_pad.clamp(min=1)