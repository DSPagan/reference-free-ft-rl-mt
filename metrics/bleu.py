"""
BLEU scoring (sacreBLEU defaults: 13a tokenization, exp smoothing). Only logged during
training as a sanity check next to chrF++; it is not a reported metric of the paper.
"""
from sacrebleu.metrics import BLEU

_bleu = BLEU()


def bleu_corpus(hyps, refs):
    """Corpus-level BLEU. hyps/refs: lists of strings, same length, same order."""
    return _bleu.corpus_score(hyps, [refs]).score
