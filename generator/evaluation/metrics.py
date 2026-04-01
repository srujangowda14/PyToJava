from typing import List
from collections import Counter
import math

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