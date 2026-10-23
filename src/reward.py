"""
The reference-free reward: a weighted hybrid of LaBSE cosine similarity and COMET-Kiwi
(source+hypothesis only, no reference used at training time).
"""
import numpy as np
import torch
from comet import download_model, load_from_checkpoint
from sentence_transformers import SentenceTransformer

from src.models import DEVICE


def load_comet_kiwi():
    """COMET-Kiwi (reference-free QE), the reward's COMET half and the training-time eval metric."""
    print("Loading COMET-Kiwi (reference-free)...")
    model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
    model.eval()
    return model


class RewardResources:
    """Loads both reward models."""

    def __init__(self):
        self.comet_kiwi = load_comet_kiwi()

        print("Loading LaBSE...")
        self.labse = SentenceTransformer("sentence-transformers/LaBSE").to(DEVICE)


class GRPORewarder:
    """
    Computes the per-sample reward for a group of K sampled hypotheses. `labse_weight`
    (w) sets the hybrid blend: reward = w*LaBSE + (1-w)*COMET-Kiwi. The paper uses an
    equal 0.5 blend.
    """

    def __init__(self, resources, labse_weight=0.5):
        self.res = resources
        self.labse_weight = labse_weight

    def get_reward(self, src_text, hyps):
        """Returns a numpy array of raw reward scores, one per hypothesis."""
        w = self.labse_weight
        return w * self._labse(src_text, hyps) + (1 - w) * self._comet_kiwi(src_text, hyps)

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
