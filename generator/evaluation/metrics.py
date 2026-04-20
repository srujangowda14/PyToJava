from typing import List
from collections import Counter
import math
import tempfile
import os
import subprocess
import re

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))

def bleu_score(
    hypothesis: List[str],
    reference:  List[str],
    max_n:      int = 4,
    smooth:     bool = True,
) -> float:
    """
    Corpus-level BLEU-4 for a single hypothesis-reference pair.
    Includes add-1 smoothing (method 1) for short sequences.
    """
    if not hypothesis:
        return 0.0
 
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(reference) / max(len(hypothesis), 1)))
 
    log_bleu = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hypothesis, n)
        ref_ngrams = _ngrams(reference,  n)
 
        clipped = sum(
            min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items()
        )
        total = max(len(hypothesis) - n + 1, 0)
 
        if smooth:
            precision = (clipped + 1) / (total + 1)
        else:
            precision = clipped / total if total > 0 else 0.0
 
        if precision == 0:
            log_bleu += float("-inf")
            break
        log_bleu += math.log(precision)
 
    return bp * math.exp(log_bleu / max_n)

def corpus_bleu_score(
    hypotheses: List[List[str]],
    references: List[List[str]],
    max_n: int = 4,
    smooth: bool = False,
) -> float:
    """Standard corpus BLEU over tokenized hypotheses and references."""
    if not hypotheses or not references:
        return 0.0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hypothesis, reference in zip(hypotheses, references):
        hyp_len += len(hypothesis)
        ref_len += len(reference)

        for n in range(1, max_n + 1):
            hyp_ngrams = _ngrams(hypothesis, n)
            ref_ngrams = _ngrams(reference, n)
            clipped_counts[n - 1] += sum(
                min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items()
            )
            total_counts[n - 1] += max(len(hypothesis) - n + 1, 0)

    if hyp_len == 0:
        return 0.0

    precisions = []
    for clipped, total in zip(clipped_counts, total_counts):
        if smooth:
            precisions.append((clipped + 1) / (total + 1))
        elif total == 0 or clipped == 0:
            return 0.0
        else:
            precisions.append(clipped / total)

    bp = 1.0 if hyp_len > ref_len else math.exp(1 - (ref_len / max(hyp_len, 1)))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)

JAVA_STRUCTURAL_KEYWORDS = {
    "class", "public", "private", "protected", "static", "final",
    "void", "return", "new", "this", "super", "extends", "implements",
    "if", "else", "for", "while", "try", "catch", "throws",
    "int", "long", "double", "float", "boolean", "String",
    "List", "Map", "ArrayList", "HashMap", "Override",
}
 
def _keyword_overlap(hyp_tokens: List[str], ref_tokens: List[str]) -> float:
    """Jaccard similarity on Java structural keyword sets."""
    hyp_kw = {t for t in hyp_tokens if t in JAVA_STRUCTURAL_KEYWORDS}
    ref_kw = {t for t in ref_tokens if t in JAVA_STRUCTURAL_KEYWORDS}
    if not ref_kw:
        return 1.0 if not hyp_kw else 0.0
    return len(hyp_kw & ref_kw) / len(hyp_kw | ref_kw)
 
 
def code_bleu(
    hypothesis: List[str],
    reference:  List[str],
    alpha:      float = 0.8,   # weight for BLEU
    beta:       float = 0.2,   # weight for keyword match
) -> float:
    """
    Lightweight CodeBLEU = alpha·BLEU + β·KeywordMatch.
 
    Full CodeBLEU also includes data-flow match; that requires
    a Java parser (e.g. tree-sitter) and is left as an extension.
    """
    bl = bleu_score(hypothesis, reference)
    kw = _keyword_overlap(hypothesis, reference)
    return alpha * bl + beta * kw

def _normalize(code: str) -> str:
    """Strip whitespace and normalize for exact match comparison."""
    return re.sub(r"\s+", " ", code).strip()
 
 
def exact_match(hypothesis: str, reference: str) -> bool:
    return _normalize(hypothesis) == _normalize(reference)
 
 
# ── Compilation check (requires javac on PATH) ────────────────────────────────
 
def check_compilable(java_code: str, class_name: str = "Translated") -> bool:
    """
    Write java_code to a temp file and attempt to compile with javac.
    Returns True if compilation succeeds.
    Requires: javac installed and on PATH.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, f"{class_name}.java")
            with open(src_path, "w") as f:
                f.write(java_code)
 
            result = subprocess.run(
                ["javac", src_path],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # javac not available — skip
        return None
    
class TranslationEvaluator:
    """
    Evaluate a batch of (hypothesis, reference) translation pairs.
 
    Usage:
        ev = TranslationEvaluator(tgt_tokenizer, tgt_vocab)
        metrics = ev.evaluate(hypotheses_ids, references_ids)
    """
 
    def __init__(self, tgt_tokenizer, tgt_vocab, check_compile: bool = False):
        self.tgt_tokenizer = tgt_tokenizer
        self.tgt_vocab     = tgt_vocab
        self.check_compile = check_compile

    def _strip_special_tokens(self, tokens: List[str]) -> List[str]:
        specials = {"<PAD>", "<SOS>", "<EOS>"}
        return [token for token in tokens if token not in specials]
 
    def evaluate(
        self,
        hyp_id_lists: List[List[int]],   # model output token id lists
        ref_id_lists: List[List[int]],   # reference token id lists
    ) -> dict:
        bleus      = []
        codebleus  = []
        exacts     = []
        compiles   = []
        corpus_hyp_tokens = []
        corpus_ref_tokens = []

        for hyp_ids, ref_ids in zip(hyp_id_lists, ref_id_lists):
            hyp_tokens = self._strip_special_tokens(self.tgt_vocab.decode(hyp_ids))
            ref_tokens = self._strip_special_tokens(self.tgt_vocab.decode(ref_ids))

            bl = bleu_score(hyp_tokens, ref_tokens)
            cb = code_bleu(hyp_tokens, ref_tokens)
 
            hyp_code = self.tgt_tokenizer.detokenize(hyp_tokens)
            ref_code = self.tgt_tokenizer.detokenize(ref_tokens)
            em = exact_match(hyp_code, ref_code)
 
            bleus.append(bl)
            codebleus.append(cb)
            exacts.append(em)
            corpus_hyp_tokens.append(hyp_tokens)
            corpus_ref_tokens.append(ref_tokens)
 
            if self.check_compile:
                ok = check_compilable(hyp_code)
                if ok is not None:
                    compiles.append(ok)
 
        metrics = {
            "bleu":       corpus_bleu_score(corpus_hyp_tokens, corpus_ref_tokens),
            "avg_sentence_bleu": sum(bleus) / len(bleus),
            "code_bleu":  sum(codebleus) / len(codebleus),
            "exact_match": sum(exacts)   / len(exacts),
            "n_samples":  len(bleus),
        }
        if compiles:
            metrics["compile_rate"] = sum(compiles) / len(compiles)
 
        return metrics
 
    def print_metrics(self, metrics: dict):
        print("\n── Evaluation Results ──────────────────────────────")
        print(f"  Samples      : {metrics['n_samples']}")
        print(f"  BLEU-4       : {metrics['bleu']:.4f}")
        print(f"  Sent BLEU    : {metrics['avg_sentence_bleu']:.4f}")
        print(f"  CodeBLEU     : {metrics['code_bleu']:.4f}")
        print(f"  Exact Match  : {metrics['exact_match']:.2%}")
        if "compile_rate" in metrics:
            print(f"  Compile Rate : {metrics['compile_rate']:.2%}")
        print("────────────────────────────────────────────────────\n")
