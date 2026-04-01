from py2java.model.seq2seq import Seq2SeqTranslator
from checkpointing.data.dataset  import (load_jsonl, build_vocabs, get_dataloaders,
                                generate_synthetic_pairs)
from py2java.model.seq2seq import Seq2SeqTranslator
from Seq2SeqTranslator.training.trainer  import Trainer
from generator.evaluation.metrics import TranslationEvaluator
from generator.utils.tokenizer   import CodeTokenizer
import random
import os
import torch
import argparse

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

def run_translate(args, config):
    import pickle
 
    vocab_path = os.path.join(os.path.dirname(args.checkpoint), "vocabs.pkl")
    with open(vocab_path, "rb") as f:
        vocabs = pickle.load(f)
    src_vocab, tgt_vocab = vocabs["src"], vocabs["tgt"]
 
    model = build_model(src_vocab, tgt_vocab, config)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
 
    # Read Python source
    with open(args.input) as f:
        py_code = f.read()
 
    src_tok  = CodeTokenizer("python")
    tgt_tok  = CodeTokenizer("java")
 
    tokens   = src_tok.tokenize(py_code)[: config["max_src_len"]]
    src_ids  = torch.tensor([src_vocab.encode(tokens)], dtype=torch.long)
    src_mask = (src_ids == src_vocab.pad_idx)
 
    if args.beam > 1:
        tgt_ids = model.translate_beam(src_ids, src_mask, beam_size=args.beam)
    else:
        tgt_ids, _ = model.translate_greedy(src_ids, src_mask)
 
    tgt_tokens = tgt_vocab.decode(tgt_ids)
    java_code  = tgt_tok.detokenize(tgt_tokens)
 
    print("\n── Generated Java ──────────────────────────────────")
    print(java_code)
    print("────────────────────────────────────────────────────")
 
    if args.output:
        with open(args.output, "w") as f:
            f.write(java_code)
        print(f"[Main] Saved to {args.output}")


def run_eval(args, config):
    import os
    import pickle
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    vocab_path = os.path.join(os.path.dirname(args.checkpoint), "vocabs.pkl")
    with open(vocab_path, "rb") as f:
        vocabs = pickle.load(f)
    src_vocab, tgt_vocab = vocabs["src"], vocabs["tgt"]

    model = build_model(src_vocab, tgt_vocab, config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pairs = load_jsonl(args.data)
    print(f"[Eval] Loaded {len(pairs)} pairs from {args.data}")

    src_tok = CodeTokenizer("python")
    tgt_tok = CodeTokenizer("java")

    hyp_ids_list = []
    ref_ids_list = []

    with torch.no_grad():
        for i, (py_code, java_code) in enumerate(pairs, 1):
            src_tokens = src_tok.tokenize(py_code)[: config["max_src_len"]]
            ref_tokens = tgt_tok.tokenize(java_code)[: config["max_tgt_len"]]

            src_ids = torch.tensor(
                [src_vocab.encode(src_tokens)],
                dtype=torch.long,
                device=device
            )
            src_mask = (src_ids == src_vocab.pad_idx)

            pred_ids, _ = model.translate_greedy(
                src_ids,
                src_mask,
                max_len=config["max_tgt_len"]
            )

            if isinstance(pred_ids, torch.Tensor):
                pred_ids = pred_ids.squeeze(0).detach().cpu().tolist()
            elif len(pred_ids) > 0 and isinstance(pred_ids[0], torch.Tensor):
                pred_ids = [x.item() for x in pred_ids]

            hyp_ids_list.append(pred_ids)
            ref_ids_list.append(tgt_vocab.encode(ref_tokens))

            if i % 100 == 0:
                print(f"[Eval] Processed {i}/{len(pairs)}")

    evaluator = TranslationEvaluator(
        tgt_tok,
        tgt_vocab,
        check_compile=args.compile_check
    )
    metrics = evaluator.evaluate(hyp_ids_list, ref_ids_list)
    evaluator.print_metrics(metrics)

def parse_args():
    p = argparse.ArgumentParser(description="Python → Java seq2seq translator")
 
    p.add_argument("--mode", choices=["train", "translate", "eval"],
                   default="train")
 
    # Data
    p.add_argument("--data",        type=str, help="Path to JSONL data file")
    p.add_argument("--val",         type=str, help="Path to validation JSONL "
                                                   "(overrides val_split)")
    p.add_argument("--synthetic",   action="store_true",
                   help="Use synthetic data for smoke-testing")
    p.add_argument("--synthetic_n", type=int, default=500,
                   help="Number of synthetic pairs to generate")
 
    # Model / training
    p.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["n_epochs"])
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--hidden_dim",  type=int,   default=DEFAULT_CONFIG["hidden_dim"])
    p.add_argument("--embed_dim",   type=int,   default=DEFAULT_CONFIG["embed_dim"])
    p.add_argument("--n_layers",    type=int,   default=DEFAULT_CONFIG["n_layers"])
    p.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--dropout",     type=float, default=DEFAULT_CONFIG["dropout"])
 
    # Checkpointing
    p.add_argument("--checkpoint",  type=str, help="Path to checkpoint (.pt)")
    p.add_argument("--resume",      type=str, help="Resume training from checkpoint")
    p.add_argument("--save_dir",    type=str, default="checkpoints")
 
    # Translation
    p.add_argument("--input",       type=str, help="Python source file to translate")
    p.add_argument("--output",      type=str, help="Output Java file path")
    p.add_argument("--beam",        type=int, default=1,
                   help="Beam size (1 = greedy)")
 
    # Evaluation
    p.add_argument("--compile_check", action="store_true",
                   help="Run javac on translated output (requires javac on PATH)")
 
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
 
    # Merge CLI args into config
    config = {**DEFAULT_CONFIG}
    config.update({
        "n_epochs":  args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "embed_dim":  args.embed_dim,
        "n_layers":   args.n_layers,
        "lr":         args.lr,
        "dropout":    args.dropout,
        "save_dir":   args.save_dir,
    })
 
    if args.mode == "train":
        run_train(args, config)
    elif args.mode == "translate":
        assert args.input and args.checkpoint, \
            "--input and --checkpoint required for translate mode"
        run_translate(args, config)
    elif args.mode == "eval":
        assert args.data and args.checkpoint, \
            "--data and --checkpoint required for eval mode"
        run_eval(args, config)
 