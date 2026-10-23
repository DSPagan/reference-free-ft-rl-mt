"""
Metrics computed on the intermediate-eval translations during training, to follow how the
model evolves. None of them enters the GRPO update, and only chrF++ picks adapters_best.

    chrf        chrF++ vs the reference (always computed)
    bleu        BLEU vs the reference
    ipd         AVLizador's dialectal purity (Catalan targets): pooled corpus-level IPD, the
                mean per-sentence IPD with unmarked sentences at 0 (what --reward ipd
                optimizes) and the western/eastern/unmarked counts
    comet       COMET-Kiwi (source + hypothesis, no reference): a fluency/adequacy check that
                the model is not unlearning to translate while chasing the reward
"""
import numpy as np
import torch

from metrics.bleu import bleu_corpus
from metrics.chrf import chrf_corpus
from src.models import DEVICE

OPTIONAL_METRICS = ("bleu", "ipd", "comet")
# Columns written to eval_metrics.csv, in order (a metric that was not computed is left empty).
COLUMNS = ["chrf", "bleu", "ipd", "ipd_mean", "ipd_western", "ipd_eastern", "ipd_unmarked", "comet_kiwi"]


def default_metrics(lang):
    """The optional metrics logged by default: bleu and comet, plus ipd for Catalan targets."""
    return ["bleu", "comet"] + (["ipd"] if lang == "cat_Latn" else [])


GAIN_KEYS = ("chrf", "bleu", "ipd", "ipd_mean", "comet_kiwi")


def format_metrics(m, signed=False):
    """Compact one-line summary of a metrics dict (as returned by EvalMetrics.compute);
    `signed` forces a +/- sign, for printing differences."""
    names = {"chrf": "chrF++", "bleu": "BLEU", "ipd": "IPD", "ipd_mean": "IPDmean", "comet_kiwi": "Kiwi"}
    spec = "+.4f" if signed else ".4f"
    return " | ".join(f"{n}: {m[k]:{spec}}" for k, n in names.items() if m.get(k) is not None)


def metric_gain(trained, vanilla):
    """trained - vanilla for every headline metric both have (None if either is missing)."""
    return {k: trained[k] - vanilla[k] for k in GAIN_KEYS
            if trained.get(k) is not None and vanilla.get(k) is not None}


class EvalMetrics:
    """`metrics` is a subset of OPTIONAL_METRICS. `comet_model` is a loaded COMET-Kiwi
    checkpoint, required when "comet" is requested."""

    def __init__(self, metrics, comet_model=None, comet_batch_size=8):
        self.metrics = set(metrics)
        self.comet_model = comet_model
        self.comet_batch_size = comet_batch_size
        if "comet" in self.metrics and comet_model is None:
            raise ValueError("the 'comet' metric needs a loaded COMET-Kiwi model")
        if "ipd" in self.metrics:
            from src.ipd_reward import corpus_ipd
            self._corpus_ipd = corpus_ipd

    def compute(self, srcs, hyps, refs):
        """Returns {metric_name: value} for the requested metrics (keys as in COLUMNS)."""
        out = {"chrf": chrf_corpus(hyps, refs)}
        if "bleu" in self.metrics:
            out["bleu"] = bleu_corpus(hyps, refs)
        if "ipd" in self.metrics:
            r = self._corpus_ipd(hyps)
            out.update(ipd=r["ipd"], ipd_mean=r["mean_sentence_ipd"], ipd_western=r["western"],
                       ipd_eastern=r["eastern"], ipd_unmarked=r["unmarked_sentences"])
        if "comet" in self.metrics:
            out["comet_kiwi"] = self._comet_kiwi(srcs, hyps)
        return out

    def _comet_kiwi(self, srcs, hyps):
        data = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
        with torch.no_grad():
            res = self.comet_model.predict(data, batch_size=self.comet_batch_size,
                                            gpus=1 if DEVICE == "cuda" else 0,
                                            num_workers=0, progress_bar=False)
            score = float(np.mean(res.scores))
        del res
        torch.cuda.empty_cache()
        return score
