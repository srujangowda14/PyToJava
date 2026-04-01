import torch
import torch.nn as nn
import math
import time
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Dict, Optional
import json
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Memory-efficient cross-entropy with label smoothing.
    logits  : [B*T, V]
    targets : [B*T]
    """
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)   # [N, V]

        mask = targets != self.pad_idx
        if mask.sum() == 0:
            return logits.new_tensor(0.0)

        log_probs = log_probs[mask]
        targets = targets[mask]

        nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
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
    
class Trainer:
    """
    Encapsulates the full training / validation loop.
 
    Usage:
        trainer = Trainer(model, train_dl, val_dl, tgt_vocab, config)
        trainer.train(n_epochs=30)
    """
 
    def __init__(
        self,
        model:      nn.Module,
        train_dl:   DataLoader,
        val_dl:     DataLoader,
        tgt_vocab,
        config:     Dict,
        device:     Optional[torch.device] = None,
    ):
        self.model    = model
        self.train_dl = train_dl
        self.val_dl   = val_dl
        self.config   = config
        self.device   = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
 
        # Loss
        self.criterion = LabelSmoothingLoss(
            vocab_size = len(tgt_vocab),
            pad_idx    = tgt_vocab.pad_idx,
            smoothing  = config.get("label_smoothing", 0.1),
        )
 
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr           = config.get("lr", 1e-3),
            weight_decay = config.get("weight_decay", 1e-5),
        )
 
        # LR schedule: warm-up + cosine annealing
        base_sched = CosineAnnealingLR(
            self.optimizer,
            T_max = config.get("n_epochs", 30),
            eta_min = 1e-6,
        )
        self.scheduler = WarmupScheduler(
            self.optimizer,
            warmup_steps  = config.get("warmup_steps", 200),
            base_scheduler = base_sched,
        )
 
        self.clip         = config.get("grad_clip", 1.0)
        self.save_dir     = config.get("save_dir", "checkpoints")
        self.best_val_loss = float("inf")
        self.history      = {"train_loss": [], "val_loss": []}
 
        os.makedirs(self.save_dir, exist_ok=True)
 
    # ── One epoch ─────────────────────────────────────────────────────────────
 
    def _run_epoch(self, dl: DataLoader, train: bool, tf_ratio: float) -> float:
        self.model.train(train)
        total_loss = 0.0
        n_batches  = 0
 
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in dl:
                src      = batch["src"].to(self.device)
                tgt      = batch["tgt"].to(self.device)
                src_mask = batch["src_mask"].to(self.device)
 
                logits = self.model(
                    src, tgt, src_mask,
                    teacher_force_ratio = tf_ratio if train else 0.0,
                )
                # logits: [B, T-1, V]  |  target: tgt[:, 1:]
                B, T, V = logits.shape
                loss = self.criterion(
                    logits.reshape(B * T, V),
                    tgt[:, 1:].reshape(B * T),
                )
 
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    self.scheduler.step()
 
                total_loss += loss.item()
                n_batches  += 1
 
        return total_loss / max(n_batches, 1)
 
    # ── Full training run ─────────────────────────────────────────────────────
 
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

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "ckpt"):
        path = os.path.join(self.save_dir, f"model_{tag}.pt")
        torch.save({
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "val_loss":   val_loss,
            "config":     self.config,
        }, path)
        print(f"  ✓ Saved checkpoint → {path}")
 
    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_val_loss = ckpt["val_loss"]
        print(f"[Trainer] Resumed from {path}  (epoch {ckpt['epoch']}, "
              f"val_loss {ckpt['val_loss']:.4f})")
 
    def _save_history(self):
        path = os.path.join(self.save_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)