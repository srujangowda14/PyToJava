# PyToJava

Python-to-Java code translation using a custom sequence-to-sequence model built in PyTorch.

This project trains a bidirectional GRU encoder with Bahdanau attention and a GRU decoder to translate Python snippets into Java snippets. The codebase includes dataset preparation, structure-aware tokenization, training, beam-search decoding, and evaluation with BLEU-style metrics.

## What This Repository Contains

- `py2java/main.py`: CLI entry point for training, translation, and evaluation
- `py2java/model/seq2seq.py`: encoder, attention module, decoder, and beam search
- `Seq2SeqTranslator/training/trainer.py`: training loop, label smoothing, scheduler, checkpointing, validation BLEU tracking
- `checkpointing/data/dataset.py`: JSONL loader, normalization, vocabulary building, batching
- `generator/utils/tokenizer.py`: structure-aware tokenizer and vocabulary
- `generator/evaluation/metrics.py`: BLEU, corpus BLEU, CodeBLEU, exact match, optional compile check
- `prepare_data.py`: converts tokenized `.py` / `.java` parallel files into JSONL files used by training

## Environment Setup

### 1. Python Version

Use Python `3.10+` if possible. The repository has been structured for modern PyTorch and standard library behavior.

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r py2java/requirements.txt
```

Current required packages:

- `torch>=2.0.0`
- `numpy>=1.24.0`

### 4. Optional Dependency for Compile Checking

If you want evaluation to also verify whether generated Java compiles, install a JDK so `javac` is available on your `PATH`.

Check it with:

```bash
javac -version
```

## Device / System Used to Run the Code

The code is written to run on either:

- CPU
- NVIDIA CUDA GPU

### How Device Selection Works

- Training uses `cuda` automatically if `torch.cuda.is_available()` is `True`; otherwise it falls back to CPU.
- Evaluation uses the same `cuda`-if-available behavior.
- Translation currently loads the checkpoint on CPU for portability.

That behavior comes directly from:

- [Seq2SeqTranslator/training/trainer.py](/Users/srujangowda/Projects/PyToJava/Seq2SeqTranslator/training/trainer.py:90)
- [py2java/main.py](/Users/srujangowda/Projects/PyToJava/py2java/main.py:171)

### Recommended Systems

- Local Linux or macOS machine with Python 3.10+
- Google Colab with an NVIDIA T4 GPU for faster training
- Any CUDA-capable GPU machine supported by your PyTorch install

### Check Which Device PyTorch Sees

```bash
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")
PY
```

## Data Format and Preparation

The training pipeline expects JSONL files where each line looks like:

```json
{"python": "def gcd ( a , b ) : NEW_LINE INDENT return a NEW_LINE DEDENT", "java": "static int gcd ( int a , int b ) { return a ; }"}
```

The repository already contains:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `train_small.jsonl`

If you need to regenerate them from tokenized parallel files in `data/`, run:

```bash
python3 prepare_data.py
```

That script reads:

- `data/train-Java-Python-tok.py`
- `data/train-Java-Python-tok.java`
- `data/val-Java-Python-tok.py`
- `data/val-Java-Python-tok.java`
- `data/test-Java-Python-tok.py`
- `data/test-Java-Python-tok.java`

and writes:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `train_small.jsonl`

## How the Tokenization Works

The tokenizer is structure-aware and lives in [generator/utils/tokenizer.py](/Users/srujangowda/Projects/PyToJava/generator/utils/tokenizer.py:39).

Important details:

- It preserves Python block structure using special tokens such as `<INDENT>`, `<DEDENT>`, and `<NL>`.
- It recognizes the repository's pretokenized dataset format and maps:
  - `NEW_LINE -> <NL>`
  - `INDENT -> <INDENT>`
  - `DEDENT -> <DEDENT>`
- It normalizes:
  - string literals to `<STR>`
  - numeric literals to `<NUM>`
- It strips comments during tokenization.

This is important because Python syntax depends on indentation, and the model needs that structural signal to learn Python-to-Java block translation.

## How to Run the Code

The project supports three modes:

- `train`
- `eval`
- `translate`

All three are run through:

```bash
python3 -m py2java.main ...
```

### 1. Train the Model

Train on the full train/validation split:

```bash
python3 -m py2java.main \
  --mode train \
  --data train.jsonl \
  --val val.jsonl \
  --save_dir checkpoints_run
```

Useful training arguments:

- `--epochs`
- `--batch_size`
- `--hidden_dim`
- `--embed_dim`
- `--n_layers`
- `--max_src_len`
- `--max_tgt_len`
- `--min_freq`
- `--seed`
- `--patience`
- `--eval_beam`
- `--beam_alpha`
- `--val_eval_max_samples`
- `--bleu_eval_interval`

Example with explicit settings:

```bash
python3 -m py2java.main \
  --mode train \
  --data train.jsonl \
  --val val.jsonl \
  --save_dir checkpoints_run \
  --epochs 30 \
  --batch_size 16 \
  --hidden_dim 256 \
  --embed_dim 128 \
  --n_layers 1 \
  --max_src_len 512 \
  --max_tgt_len 768 \
  --eval_beam 4 \
  --beam_alpha 0.6
```

#### Quick Smoke Test

Synthetic-data smoke test:

```bash
python3 -m py2java.main \
  --mode train \
  --synthetic \
  --synthetic_n 500 \
  --epochs 3 \
  --save_dir checkpoints_smoke
```

Small-subset run:

```bash
python3 -m py2java.main \
  --mode train \
  --data train_small.jsonl \
  --val val.jsonl \
  --save_dir checkpoints_small
```

#### Resume Training

```bash
python3 -m py2java.main \
  --mode train \
  --data train.jsonl \
  --val val.jsonl \
  --resume checkpoints_run/model_best.pt \
  --save_dir checkpoints_run
```

### 2. Evaluate a Checkpoint

Evaluate on the held-out test set:

```bash
python3 -m py2java.main \
  --mode eval \
  --data test.jsonl \
  --checkpoint checkpoints_run/model_best_bleu.pt \
  --beam 4
```

Optional compile check:

```bash
python3 -m py2java.main \
  --mode eval \
  --data test.jsonl \
  --checkpoint checkpoints_run/model_best_bleu.pt \
  --beam 4 \
  --compile_check
```

### 3. Translate a Single Input File

```bash
python3 -m py2java.main \
  --mode translate \
  --input sample.py \
  --checkpoint checkpoints_run/model_best_bleu.pt \
  --beam 4 \
  --output sample_translated.java
```

Notes:

- The model is trained on tokenized / normalized code-like inputs, so best results come from inputs that resemble the dataset format.
- The tokenizer can still process raw source text, but output quality depends on how closely the input matches the training distribution.

## How Results Are Generated

This section explains exactly how the reported metrics are produced in this repository.

### Step 1. Load and Normalize Data

Training, validation, and test files are loaded through `load_jsonl()` in [checkpointing/data/dataset.py](/Users/srujangowda/Projects/PyToJava/checkpointing/data/dataset.py:126).

That loader:

- reads JSONL pairs from disk
- normalizes whitespace
- skips empty pairs
- optionally deduplicates pairs during training

### Step 2. Build Vocabularies

`build_vocabs()` in [checkpointing/data/dataset.py](/Users/srujangowda/Projects/PyToJava/checkpointing/data/dataset.py:136) tokenizes the training pairs and builds:

- source vocabulary for Python tokens
- target vocabulary for Java tokens

Tokens that appear fewer than `min_freq` times are mapped to `<UNK>`.

### Step 3. Create Batches

`get_dataloaders()` in [checkpointing/data/dataset.py](/Users/srujangowda/Projects/PyToJava/checkpointing/data/dataset.py:151) creates PyTorch dataloaders.

The pipeline:

- encodes source and target token sequences
- adds `<SOS>` and `<EOS>` to targets
- pads variable-length batches
- creates source padding masks
- uses bucketed batching during training to group similar-length examples

### Step 4. Train the Model

Training happens in [Seq2SeqTranslator/training/trainer.py](/Users/srujangowda/Projects/PyToJava/Seq2SeqTranslator/training/trainer.py:211).

During training, the project uses:

- teacher forcing with linear decay from `1.0` to `0.5`
- label smoothing
- gradient clipping
- Adam optimizer
- linear warm-up
- cosine annealing

Two kinds of checkpoints are saved:

- `model_best.pt`: best validation loss
- `model_best_bleu.pt`: best validation BLEU

This means the final "best model" for translation quality is selected on the validation set, not the test set.

### Step 5. Decode Predictions

At evaluation time, the model generates predictions using:

- greedy decoding if `--beam 1`
- beam search if `--beam > 1`

Beam search uses length normalization, and optionally supports compile-aware reranking during inference.

### Step 6. Compute Metrics

Metrics are computed in [generator/evaluation/metrics.py](/Users/srujangowda/Projects/PyToJava/generator/evaluation/metrics.py:155).

The evaluation reports:

- `BLEU-4`: corpus BLEU across the full evaluation set
- `Sent BLEU`: average sentence-level BLEU across examples
- `CodeBLEU`: lightweight BLEU plus Java keyword overlap
- `Exact Match`: normalized string equality after detokenization
- `Compile Rate`: optional, if `--compile_check` is enabled and `javac` is installed

### Important Interpretation Note

The dataset in this repository is already tokenized and normalized, so the reported scores reflect performance on that processed representation. Results on raw hand-written Python files may differ unless the raw input is preprocessed to match the same format.

## Model Architecture Summary

The actual model implementation is in [py2java/model/seq2seq.py](/Users/srujangowda/Projects/PyToJava/py2java/model/seq2seq.py:6).

Architecture:

- bidirectional GRU encoder
- Bahdanau additive attention
- GRU decoder
- input feeding in the decoder
- greedy and beam-search decoding

Default configuration from [py2java/main.py](/Users/srujangowda/Projects/PyToJava/py2java/main.py:12):

- `embed_dim = 128`
- `hidden_dim = 256`
- `n_layers = 1`
- `dropout = 0.3`
- `batch_size = 16`
- `epochs = 30`

## Example Workflow

Train:

```bash
python3 -m py2java.main \
  --mode train \
  --data train.jsonl \
  --val val.jsonl \
  --save_dir checkpoints_demo
```

Evaluate best BLEU checkpoint:

```bash
python3 -m py2java.main \
  --mode eval \
  --data test.jsonl \
  --checkpoint checkpoints_demo/model_best_bleu.pt \
  --beam 4
```

Translate a file:

```bash
python3 -m py2java.main \
  --mode translate \
  --input sample.py \
  --checkpoint checkpoints_demo/model_best_bleu.pt \
  --beam 4 \
  --output sample_translated.java
```

## Running on Google Colab

Basic Colab flow:

```python
from google.colab import files
uploaded = files.upload()
```

```bash
%%bash
unzip -q PyToJava_colab.zip -d /content/PyToJava
cd /content/PyToJava
pip install -q -r py2java/requirements.txt
```

GPU check:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Training command:

```bash
%%bash
cd /content/PyToJava
python3 -m py2java.main \
  --mode train \
  --data train.jsonl \
  --val val.jsonl \
  --save_dir checkpoints_colab
```

If you are using a T4 and run into memory pressure, reduce:

- `--batch_size`
- `--max_src_len`
- `--max_tgt_len`

## Troubleshooting

### `ModuleNotFoundError: No module named 'py2java'`

Run the command from the repository root:

```bash
cd /path/to/PyToJava
python3 -m py2java.main --help
```

In Colab, put `cd` and `python3 -m ...` in the same `%%bash` cell.

### Training Is Very Slow

This is expected on CPU. Use a CUDA GPU for reasonable training speed.

### CUDA Out of Memory

Try smaller settings, for example:

```bash
python3 -m py2java.main \
  --mode train \
  --data train.jsonl \
  --val val.jsonl \
  --save_dir checkpoints_small_gpu \
  --batch_size 2 \
  --max_src_len 128 \
  --max_tgt_len 192
```

### `javac` Compile Check Does Not Run

Install a JDK and make sure `javac` is on your `PATH`.

## Summary

To reproduce results in this repository:

1. install the Python dependencies
2. prepare or reuse the JSONL dataset files
3. train with `python3 -m py2java.main --mode train ...`
4. evaluate with `python3 -m py2java.main --mode eval ...`
5. use `model_best_bleu.pt` when you want the checkpoint chosen by validation BLEU

This README is aligned with the current repository behavior, including data loading, tokenization, training, checkpoint selection, and evaluation.
