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
