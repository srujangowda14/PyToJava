from model.seq2seq import Seq2SeqTranslator

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