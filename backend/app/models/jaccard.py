import numpy as np

from ..preprocessing import preprocess
from .base import RetrievalModel


class JaccardModel(RetrievalModel):
    """Vectores binarios: sim(Q, D) = |Q ∩ D| / |Q ∪ D| = c / (|Q| + |D| - c)."""

    name = "jaccard"

    def score_all(self, query: str) -> np.ndarray:
        idx = self.index
        q_terms = set(preprocess(query))
        common = np.zeros(idx.n_docs, dtype=np.float64)
        for t in q_terms:
            if t in idx.postings:
                common[idx.postings[t][0]] += 1
        union = len(q_terms) + idx.doc_nterms - common
        scores = np.zeros_like(common)
        mask = common > 0
        scores[mask] = common[mask] / union[mask]
        return scores
