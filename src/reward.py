"""
The reference-free reward: a per-language-weighted hybrid of LaBSE cosine similarity and
COMET-Kiwi (source+hypothesis only, no reference used at training time). Also supports
"semantic" (LaBSE only) and "comet" (COMET-Kiwi only) for the reward-component ablation,
and "random" (i.i.d. uniform noise, uncorrelated with quality) as a control for whether
GRPO's effect is reward-content-driven or an artifact of the optimization loop itself.
"""
import numpy as np
import torch
from comet import download_model, load_from_checkpoint
from sentence_transformers import SentenceTransformer

from src.models import DEVICE

REWARD_MODES = ("hybrid", "semantic", "comet", "random")


class RewardResources:
    """Loads only what the chosen reward mode actually needs."""

    def __init__(self, reward_mode, model_name):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.comet_kiwi = None
        if reward_mode in ("hybrid", "comet"):
            print("Loading COMET-Kiwi (reward model, reference-free)...")
            self.comet_kiwi = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
            self.comet_kiwi.eval()

        self.labse = None
        if reward_mode in ("hybrid", "semantic"):
            print("Loading LaBSE...")
            self.labse = SentenceTransformer("sentence-transformers/LaBSE").to(DEVICE)


class GRPORewarder:
    """
    Computes the per-sample reward for a group of K sampled hypotheses. `labse_weight`
    (w) sets the hybrid blend: reward = w*LaBSE + (1-w)*COMET-Kiwi. The main sweep uses a
    per-language calibrated weight (see reward_weights.csv / src/calibration.py), not a
    flat 0.5.
    """

    def __init__(self, resources, mode, labse_weight=0.5):
        assert mode in REWARD_MODES, f"unknown reward mode: {mode}"
        self.res = resources
        self.mode = mode
        self.labse_weight = labse_weight

    def get_reward(self, src_text, hyps):
        """Returns a numpy array of raw reward scores, one per hypothesis."""
        if self.mode == "random":
            return np.random.rand(len(hyps)).astype(np.float32)
        if self.mode == "hybrid":
            w = self.labse_weight
            return w * self._labse(src_text, hyps) + (1 - w) * self._comet_kiwi(src_text, hyps)
        if self.mode == "semantic":
            return self._labse(src_text, hyps)
        if self.mode == "comet":
            return self._comet_kiwi(src_text, hyps)

    def _labse(self, src_text, hyps):
        with torch.no_grad():
            embs = self.res.labse.encode([src_text] + hyps, convert_to_tensor=True, normalize_embeddings=True)
            scores = (embs[0] @ embs[1:].T).cpu().numpy()
        del embs
        torch.cuda.empty_cache()
        return scores

    def _comet_kiwi(self, src_text, hyps):
        data = [{"src": src_text, "mt": h} for h in hyps]
        with torch.no_grad():
            out = self.res.comet_kiwi.predict(data, batch_size=len(hyps),
                                                gpus=1 if DEVICE == "cuda" else 0,
                                                num_workers=0, progress_bar=False)
            scores = np.array(out.scores)
        del out
        torch.cuda.empty_cache()
        return scores
