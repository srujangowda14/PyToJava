from generator.utils.tokenizer import (
    CodeTokenizer, Vocabulary,
    SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
)
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import json

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

def load_jsonl(path: str) -> List[Tuple[str, str]]:
    """Load JSONL file with {"python": ..., "java": ...} records."""
    pairs = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["python"], obj["java"]))
    print(f"[Data] Loaded {len(pairs)} pairs from {path}")
    return pairs
 
 
def build_vocabs(
    pairs: List[Tuple[str, str]],
    min_freq: int = 2,
) -> Tuple[Vocabulary, Vocabulary]:
    """Build source (Python) and target (Java) vocabularies from parallel pairs."""
    src_tok = CodeTokenizer("python")
    tgt_tok = CodeTokenizer("java")
 
    src_token_lists = [src_tok.tokenize(py)   for py, _  in pairs]
    tgt_token_lists = [tgt_tok.tokenize(java) for _,  java in pairs]
 
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build(src_token_lists, min_freq=min_freq)
    tgt_vocab.build(tgt_token_lists, min_freq=min_freq)
 
    return src_vocab, tgt_vocab

def get_dataloaders(
    train_pairs: List[Tuple[str, str]],
    val_pairs:   List[Tuple[str, str]],
    src_vocab:   Vocabulary,
    tgt_vocab:   Vocabulary,
    batch_size:  int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
 
    src_tok = CodeTokenizer("python")
    tgt_tok = CodeTokenizer("java")
 
    from functools import partial
    _collate = partial(
        collate_fn,
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
    )
 
    train_ds = CodeTranslationDataset(train_pairs, src_vocab, tgt_vocab, src_tok, tgt_tok)
    val_ds   = CodeTranslationDataset(val_pairs,   src_vocab, tgt_vocab, src_tok, tgt_tok)
 
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=_collate, num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          collate_fn=_collate, num_workers=num_workers)
 
    return train_dl, val_dl

PYTHON_TEMPLATES = [
    """\
class {name}:
    def __init__(self, value):
        self.value = value
 
    def get_value(self):
        return self.value
 
    def set_value(self, value):
        self.value = value
 
    def __str__(self):
        return str(self.value)
""",
    """\
class {name}:
    def __init__(self, items):
        self.items = items
 
    def add(self, item):
        self.items.append(item)
 
    def remove(self, item):
        self.items.remove(item)
 
    def size(self):
        return len(self.items)
 
    def contains(self, item):
        return item in self.items
""",
    """\
class {name}:
    count = 0
 
    def __init__(self, name, age):
        self.name = name
        self.age = age
        {name}.count += 1
 
    def greet(self):
        return "Hello, " + self.name
 
    def is_adult(self):
        return self.age >= 18
 
    @staticmethod
    def get_count():
        return {name}.count
""",
]
 
JAVA_TEMPLATES = [
    """\
public class {name} {{
    private int value;
 
    public {name}(int value) {{
        this.value = value;
    }}
 
    public int getValue() {{
        return value;
    }}
 
    public void setValue(int value) {{
        this.value = value;
    }}
 
    @Override
    public String toString() {{
        return String.valueOf(value);
    }}
}}
""",
    """\
import java.util.ArrayList;
import java.util.List;
 
public class {name} {{
    private List<Object> items;
 
    public {name}(List<Object> items) {{
        this.items = new ArrayList<>(items);
    }}
 
    public void add(Object item) {{
        items.add(item);
    }}
 
    public void remove(Object item) {{
        items.remove(item);
    }}
 
    public int size() {{
        return items.size();
    }}
 
    public boolean contains(Object item) {{
        return items.contains(item);
    }}
}}
""",
    """\
public class {name} {{
    private static int count = 0;
    private String name;
    private int age;
 
    public {name}(String name, int age) {{
        this.name = name;
        this.age = age;
        {name}.count++;
    }}
 
    public String greet() {{
        return "Hello, " + name;
    }}
 
    public boolean isAdult() {{
        return age >= 18;
    }}
 
    public static int getCount() {{
        return count;
    }}
}}
""",
]

CLASS_NAMES = [
    "Counter", "Container", "Person", "Node", "Stack",
    "Queue", "Calculator", "Manager", "Handler", "Processor",
    "Validator", "Parser", "Builder", "Factory", "Wrapper",
]
 
 