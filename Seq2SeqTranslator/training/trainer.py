import torch
import torch.nn as nn
import math
import time

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
    
class WarmupScheduler:
    """
    Linear warm-up for `warmup_steps` then delegates to base scheduler.
    """
 
    def __init__(self, optimizer, warmup_steps: int, base_scheduler):
        self.optimizer      = optimizer
        self.warmup_steps   = warmup_steps
        self.base_scheduler = base_scheduler
        self.step_count     = 0
        self._base_lrs      = [g["lr"] for g in optimizer.param_groups]
 
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale
        else:
            self.base_scheduler.step()
 
    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]
    
def train(self, n_epochs: int):
        print(f"\n[Trainer] Device: {self.device}")
        print(f"[Trainer] Model parameters: {self.model.count_parameters():,}\n")
 
        for epoch in range(1, n_epochs + 1):
            # Teacher forcing ratio: linearly decay 1.0 → 0.5 over training
            tf_ratio = max(0.5, 1.0 - (epoch / n_epochs) * 0.5)
 
            t0 = time.time()
            train_loss = self._run_epoch(self.train_dl, train=True,  tf_ratio=tf_ratio)
            val_loss   = self._run_epoch(self.val_dl,   train=False, tf_ratio=0.0)
            elapsed    = time.time() - t0
 
            train_ppl = math.exp(min(train_loss, 20))
            val_ppl   = math.exp(min(val_loss,   20))
            lr        = self.scheduler.get_last_lr()[0]
 
            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"Train Loss {train_loss:.4f} (PPL {train_ppl:.1f}) | "
                f"Val Loss {val_loss:.4f} (PPL {val_ppl:.1f}) | "
                f"LR {lr:.2e} | TF {tf_ratio:.2f} | "
                f"Time {elapsed:.1f}s"
            )
 
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
 
            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, tag="best")
 
            # Periodic checkpoint every 5 epochs
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, val_loss, tag=f"epoch{epoch}")
 
        self._save_history()
        print(f"\n[Trainer] Training complete. Best val loss: {self.best_val_loss:.4f}")