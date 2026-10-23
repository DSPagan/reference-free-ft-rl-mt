"""
Dialectal-purity reward: the Índex de Puresa Dialectal (IPD) from AVLizador
(third_party/AVLizador, git submodule), i.e. the share of dialectally marked words in a
Catalan text that are Western/Valencian (AVL) forms rather than Eastern Catalan ones:

    IPD = western / (western + eastern)

1.0 is fully Valencian, 0.0 fully Eastern Catalan. Texts with no marked word get IPD
None, which the reward maps to `none_value` (0.0 by default, so unmarked output is not
rewarded). The reward only looks at the hypothesis, never at the source.
"""
import sys
from pathlib import Path

import numpy as np

_IPD_DIR = Path(__file__).resolve().parent.parent / "third_party" / "AVLizador" / "pipeline" / "09_index_puresa_dialectal"


def _load_ipd():
    """Imports AVLizador's IPD module and its dialect index; fails loudly (AVLizador itself
    silently skips POS filtering) if the submodule or the spaCy Catalan model is missing."""
    if not (_IPD_DIR / "ipd.py").exists():
        raise FileNotFoundError(f"{_IPD_DIR}/ipd.py not found -- run `git submodule update --init`")
    if str(_IPD_DIR) not in sys.path:
        sys.path.insert(0, str(_IPD_DIR))
    import ipd
    from traductor.rules import pos_tagger

    if pos_tagger._carrega_spacy() is None:
        raise RuntimeError("spaCy Catalan model not loadable (ca_core_news_md); run `uv sync`")
    return ipd.calcula_ipd, ipd.carrega_formes_dialectals()


class IPDRewarder:
    """Per-hypothesis IPD reward for a GRPO group, same interface as GRPORewarder."""

    def __init__(self, none_value=0.0):
        self.none_value = none_value
        self._calcula_ipd, self._formes = _load_ipd()

    def get_reward(self, src_text, hyps):
        """Returns a numpy array of IPD scores, one per hypothesis (src_text is unused)."""
        scores = [self._calcula_ipd(h, self._formes)["ipd"] for h in hyps]
        return np.array([self.none_value if s is None else s for s in scores], dtype=np.float32)


def corpus_ipd(texts):
    """Corpus-level IPD for evaluation: western / (western + eastern) pooled over all texts,
    plus the number of texts with no marked word. Returns a dict."""
    calcula_ipd, formes = _load_ipd()
    occ = ori = unmarked = 0
    for t in texts:
        r = calcula_ipd(t, formes)
        occ += r["total_occidentals"]
        ori += r["total_orientals"]
        unmarked += r["ipd"] is None
    return {"ipd": occ / (occ + ori) if occ + ori else None, "western": occ, "eastern": ori,
            "unmarked_sentences": unmarked, "sentences": len(texts)}
