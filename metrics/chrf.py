"""
chrF++ scoring (the paper's primary metric). Word order 2 everywhere, matching the
protocol used for every reported number.
"""
from sacrebleu.metrics import CHRF

_chrf = CHRF(word_order=2)


def chrf_corpus(hyps, refs):
    """Corpus-level chrF++. hyps/refs: lists of strings, same length, same order."""
    return _chrf.corpus_score(hyps, [refs]).score


def chrf_sentence(hyp, ref):
    """Sentence-level chrF++ for a single hypothesis/reference pair."""
    return _chrf.sentence_score(hyp, [ref]).score
