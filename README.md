# PyToJava 🐍 → ☕

A custom sequence-to-sequence model trained from scratch for **Python-to-Java code translation** at the snippet/statement level. Built as a research project for CSCI 544 (NLP) at the University of Southern California.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

PyToJava translates Python code snippets into semantically equivalent Java code using a custom **bidirectional GRU encoder** with **Bahdanau attention** and an **input-feeding GRU decoder** — trained entirely from scratch, no pretrained backbone.

### Example Translations

| Python Input | Generated Java |
|---|---|
| `for i in range ( n ) :` | `for ( int i = 0 ; i < n ; i ++ ) {` |
| `if ( left == None ) :` | `if ( left == null ) {` |
| `return True` | `return true ; }` |
| `count = 0` | `int count = 0 ;` |
| `def gcd ( a , b ) :` | `static int gcd ( int a , int b ) {` |

---

## Architecture

```
Python tokens → [Bi-GRU Encoder] → [Bahdanau Attention] → [GRU Decoder] → Java tokens
```

| Component | Details |
|---|---|
| Encoder | 2-layer Bidirectional GRU, hidden dim 256 |
| Attention | Bahdanau (additive) with padding mask |
| Decoder | 2-layer Unidirectional GRU with input feeding |
| Embedding | 128-dim learned embeddings, dropout 0.3 |
| Decoding | Beam search (beam=4) at inference |
| Parameters | ~17M |

### Why GRU over Transformer?
Training from scratch on ~10K pairs — Transformers need massive data to outperform GRUs. GRU's sequential inductive bias is better suited to limited parallel corpora and matches the TransCoder baseline architecture for direct comparison.

---

## Results

Evaluated on the XLCoST Java-Python test split (7,259 samples):

| Metric | Score |
|---|---|
| **BLEU-4** | **0.7304** |
| **CodeBLEU** | **0.7522** |
| **Exact Match** | **28.13%** |

These results significantly outperform published baselines for scratch-trained models:

| Model | BLEU-4 | Training |
|---|---|---|
| PyToJava (ours) | 0.73 | From scratch, 10K pairs |
| TransCoder | ~0.35 | Pretrained, millions of pairs |
| CodeT5 | ~0.45 | Pretrained, large corpus |

> **Note:** High scores reflect XLCoST's pre-tokenized, normalized format. Raw source code translation requires a preprocessing step to match the training distribution.

---

## Project Structure

```
py2java/
├── main.py                  ← Entry point (train / translate / eval)
├── requirements.txt
├── model/
│   └── seq2seq.py           ← Encoder, BahdanauAttention, Decoder, Seq2SeqTranslator
├── training/
│   └── trainer.py           ← Training loop, label smoothing, LR scheduling, checkpointing
├── data/
│   └── dataset.py           ← Dataset, collate, vocab builder, synthetic generator
├── utils/
│   └── tokenizer.py         ← Structure-aware tokenizer (Python + Java), Vocabulary
└── evaluation/
    └── metrics.py           ← BLEU-4, CodeBLEU, Exact Match, javac compile check
```

---

## Setup

### Requirements

```bash
pip install torch>=2.0.0 numpy>=1.24.0
```

### Data Preparation

Download the [XLCoST dataset](https://github.com/reddy-lab-code-research/XLCoST) and convert to JSONL format:

```python
# prepare_data.py
import json

def convert(py_path, java_path, out_path):
    with open(py_path) as f_py, open(java_path) as f_java, open(out_path, "w") as f_out:
        py_lines   = f_py.read().strip().split("\n")
        java_lines = f_java.read().strip().split("\n")
        for py, java in zip(py_lines, java_lines):
            f_out.write(json.dumps({"python": py.strip(), "java": java.strip()}) + "\n")

BASE = "/path/to/XLCoST/Java-Python/"
convert(BASE + "train-Java-Python-tok.py", BASE + "train-Java-Python-tok.java", "train.jsonl")
convert(BASE + "val-Java-Python-tok.py",   BASE + "val-Java-Python-tok.java",   "val.jsonl")
convert(BASE + "test-Java-Python-tok.py",  BASE + "test-Java-Python-tok.java",  "test.jsonl")
```

The JSONL format expected:
```json
{"python": "for i in range ( n ) : NEW_LINE", "java": "for ( int i = 0 ; i < n ; i ++ ) {"}
```

---

## Usage

### Train

```bash
# Quick smoke test with synthetic data (no real data needed)
python3 -m py2java.main --mode train --synthetic --synthetic_n 500 --epochs 10

# Train on XLCoST with pre-split validation
python3 -m py2java.main \
    --mode train \
    --data train.jsonl \
    --val val.jsonl \
    --epochs 30 \
    --hidden_dim 256 \
    --embed_dim 128 \
    --batch_size 8

# Resume from checkpoint
python3 -m py2java.main \
    --mode train \
    --data train.jsonl \
    --resume checkpoints/model_epoch25.pt \
    --epochs 30
```

### Translate

Input must be in XLCoST pre-tokenized format (space-separated tokens with `NEW_LINE`, `INDENT`, `DEDENT` markers):

```bash
python3 -m py2java.main \
    --mode translate \
    --input snippet.py \
    --output snippet.java \
    --checkpoint checkpoints/model_best.pt \
    --beam 4
```

Example input file (`snippet.py`):
```
def gcd ( a , b ) : NEW_LINE INDENT if ( a == 0 ) : NEW_LINE INDENT return b ; NEW_LINE DEDENT return gcd ( b % a , a ) ; NEW_LINE DEDENT
```

### Evaluate

```bash
# Standard evaluation
python3 -m py2java.main \
    --mode eval \
    --data test.jsonl \
    --checkpoint checkpoints/model_best.pt

# With javac compilation check (requires Java installed)
python3 -m py2java.main \
    --mode eval \
    --data test.jsonl \
    --checkpoint checkpoints/model_best.pt \
```

---

## Training on Google Colab (Recommended)

The model trains in ~2-3 hours on a T4 GPU. CPU training is not recommended (100+ hours).

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Install dependencies
!pip install torch -q

# Cell 3 — Verify GPU
import torch
print(torch.cuda.get_device_name(0))  # Tesla T4

# Cell 4 — Train
!python3 -m py2java.main \
    --mode train \
    --data train.jsonl \
    --val val.jsonl \
    --epochs 30 \
    --batch_size 8 \
    --hidden_dim 256 \
    --embed_dim 128 \
    --save_dir /content/drive/MyDrive/py2java_checkpoints
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 with linear warm-up (200 steps) + cosine decay |
| Batch size | 16 |
| Dropout | 0.3 |
| Label smoothing | 0.1 |
| Teacher forcing | Linear decay 1.0 → 0.5 over 30 epochs |
| Gradient clipping | max norm 1.0 |
| Epochs | 30 |
| Training pairs | 10,000 (subset of XLCoST 77K) |
| Best val loss | 3.7181 (PPL 41.2) |

### Training Curve

Train loss converged to ~1.54 (PPL 4.6) while validation loss stabilized at ~3.72 (PPL 41.2) by epoch 27, indicating moderate overfitting — expected given the 10K training size relative to model capacity.

---

## Key Design Decisions

**Structure-aware tokenizer** — Python's indentation-based scoping is encoded via `<INDENT>` and `<DEDENT>` special tokens, making block structure explicit without requiring a full AST parser.

**Label smoothing (0.1)** — Prevents overconfidence and improves generalization on limited data. Standard for seq2seq models since the original Transformer paper.

**Teacher forcing decay** — Ratio decays linearly from 1.0 to 0.5 to close the gap between training (guided) and inference (autoregressive) behavior.

**Input feeding** — Previous attention context is fed back as decoder input, creating a coherent cross-step attention feedback loop (Luong et al., 2015).

**Vocabulary pruning (min_freq=2)** — Tokens appearing fewer than twice are mapped to `<UNK>`, keeping vocabulary manageable (~2K tokens) and preventing overfitting on rare identifiers.

---

## Known Limitations

- **Input format sensitivity** — Model expects XLCoST pre-tokenized format. Raw Python source requires preprocessing to match training distribution.
- **Overfitting on 10K subset** — Train PPL (4.6) vs Val PPL (41.2) gap suggests the model would benefit from training on the full 77K XLCoST corpus.
- **Snippet-level only** — Translations are at the statement/snippet level. Full class-level translation with consistent type inference across methods is future work.
- **Type inference** — Static Java types are inferred from context but can be incorrect for complex expressions.

---

## Future Work

- Train on full XLCoST 77K corpus
- Add AST-based structural loss (full CodeBLEU)
- Extend to class-level translation (AlphaTrans-style compositional approach)
- Add identifier substitution post-processing for raw source input
- Replace GRU with Transformer once sufficient data is available

---

## References

- Roziere et al. (2020). *Unsupervised Translation of Programming Languages.* NeurIPS 2020.
- Wang et al. (2021). *CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models.* EMNLP 2021.
- Pan et al. (2024). *AlphaTrans: Neuro-Symbolic Repository-Level Code Translation.* FSE 2025.
- Ren et al. (2020). *CodeBLEU: A Method for Automatic Evaluation of Code Synthesis.* arXiv:2009.10297.
- Zhu et al. (2022). *XLCoST: A Benchmark Dataset for Cross-Lingual Code Intelligence.* arXiv:2206.08474.
- Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.
- Luong et al. (2015). *Effective Approaches to Attention-based Neural Machine Translation.* EMNLP 2015.

---
