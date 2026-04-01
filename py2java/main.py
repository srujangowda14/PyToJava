from model.seq2seq import Seq2SeqTranslator
from checkpointing.data.dataset      import (load_jsonl, build_vocabs, get_dataloaders,
                                generate_synthetic_pairs)
from model.seq2seq     import Seq2SeqTranslator
from Seq2SeqTranslator.training.trainer  import Trainer
from generator.evaluation.metrics import TranslationEvaluator
from generator.utils.tokenizer   import CodeTokenizer
import random
import os

DEFAULT_CONFIG = {
    # Model
    "embed_dim":       256,
    "hidden_dim":      512,
    "n_layers":        2,
    "dropout":         0.3,
    # Training
    "lr":              1e-3,
    "weight_decay":    1e-5,
    "grad_clip":       1.0,
    "label_smoothing": 0.1,
    "warmup_steps":    200,
    "batch_size":      16,
    "n_epochs":        30,
    # Data
    "max_src_len":     512,
    "max_tgt_len":     768,
    "min_freq":        2,
    "val_split":       0.1,
    # I/O
    "save_dir":        "checkpoints",
}
 
 
 
def build_model(src_vocab, tgt_vocab, config) -> Seq2SeqTranslator:
    return Seq2SeqTranslator(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        embed_dim      = config["embed_dim"],
        hidden_dim     = config["hidden_dim"],
        n_layers       = config["n_layers"],
        dropout        = config["dropout"],
        src_pad_idx    = src_vocab.pad_idx,
        tgt_pad_idx    = tgt_vocab.pad_idx,
        sos_idx        = tgt_vocab.sos_idx,
        eos_idx        = tgt_vocab.eos_idx,
    )

def run_train(args, config):
    # Load data
    if args.synthetic:
        pairs = generate_synthetic_pairs(n=args.synthetic_n)
    else:
        pairs = load_jsonl(args.data)
 
    # Train / val split
    random.shuffle(pairs)
    split = int(len(pairs) * (1 - config["val_split"]))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
 
    # Vocabularies
    src_vocab, tgt_vocab = build_vocabs(train_pairs, min_freq=config["min_freq"])
    print(f"[Main] Src vocab: {len(src_vocab)}  |  Tgt vocab: {len(tgt_vocab)}")
 
    # DataLoaders
    train_dl, val_dl = get_dataloaders(
        train_pairs, val_pairs,
        src_vocab, tgt_vocab,
        batch_size = config["batch_size"],
    )
 
    # Model
    model = build_model(src_vocab, tgt_vocab, config)
    print(f"[Main] Model parameters: {model.count_parameters():,}")
 
    # Trainer
    trainer = Trainer(model, train_dl, val_dl, tgt_vocab, config)
 
    if args.resume:
        trainer.load_checkpoint(args.resume)
 
    trainer.train(n_epochs=config["n_epochs"])
 
    # Save vocab for later use
    import pickle
    os.makedirs(config["save_dir"], exist_ok=True)
    with open(os.path.join(config["save_dir"], "vocabs.pkl"), "wb") as f:
        pickle.dump({"src": src_vocab, "tgt": tgt_vocab}, f)
    print("[Main] Vocabs saved.")