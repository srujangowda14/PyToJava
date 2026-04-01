from generator.utils.tokenizer import (
    CodeTokenizer, Vocabulary,
    SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
)
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset

class CodeTranslationDataset(Dataset):
    """
    PyTorch Dataset for Python→Java class-level translation.
 
    Each item is a (src_ids, tgt_ids) pair of integer tensors
    with SOS/EOS boundaries on the target side.
    """
 
    def __init__(
        self,
        pairs:    List[Tuple[str, str]],   # [(python_code, java_code), ...]
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        src_tokenizer: CodeTokenizer,
        tgt_tokenizer: CodeTokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 768,
    ):
        self.pairs         = pairs
        self.src_vocab     = src_vocab
        self.tgt_vocab     = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_len   = max_src_len
        self.max_tgt_len   = max_tgt_len
 
    def __len__(self):
        return len(self.pairs)
 
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        py_code, java_code = self.pairs[idx]
 
        # tokenize
        src_tokens = self.src_tokenizer.tokenize(py_code)[: self.max_src_len]
        tgt_tokens = self.tgt_tokenizer.tokenize(java_code)[: self.max_tgt_len - 2]
 
        # encode to ids
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = (
            [self.tgt_vocab.sos_idx]
            + self.tgt_vocab.encode(tgt_tokens)
            + [self.tgt_vocab.eos_idx]
        )
 
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }
    
def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    src_pad_idx: int,
    tgt_pad_idx: int,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences to the same length."""
    src_seqs = [item["src"] for item in batch]
    tgt_seqs = [item["tgt"] for item in batch]
 
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_seqs, batch_first=True, padding_value=src_pad_idx
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_seqs, batch_first=True, padding_value=tgt_pad_idx
    )
 
    # Boolean padding masks  (True = ignore this position)
    src_mask = (src_padded == src_pad_idx)
    tgt_mask = (tgt_padded == tgt_pad_idx)
 
    return {
        "src":      src_padded,
        "tgt":      tgt_padded,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }
 