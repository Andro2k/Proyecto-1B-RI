import math
from collections import Counter

import numpy as np

from ..inverted_index import InvertedIndex
from ..preprocessing import preprocess
from .base import RetrievalModel


class TfidfModel(RetrievalModel):
    """Coseno con pesos w = (1 + log10 tf) * log10(N / df), tanto en documentos como en consultas."""

    name = "tfidf"

    def __init__(self, index: InvertedIndex) -> None:
        super().__init__(index)
        n = index.n_docs
        self.idf = {t: math.log10(n / len(p[0])) for t, p in index.postings.items()}
        # Norma euclídea de cada documento (se calcula una sola vez recorriendo los postings)
        sq = np.zeros(n, dtype=np.float64)
        for t, (docs, tfs) in index.postings.items():
            w = (1.0 + np.log10(tfs)) * self.idf[t]
            sq[docs] += w * w
        self.doc_norm = np.sqrt(sq)

    def score_all(self, query: str) -> np.ndarray:
        idx = self.index
        q_tf = Counter(t for t in preprocess(query) if t in idx.postings)
        dot = np.zeros(idx.n_docs, dtype=np.float64)
        q_sq = 0.0
        for t, tf in q_tf.items():
            wq = (1.0 + math.log10(tf)) * self.idf[t]
            q_sq += wq * wq
            docs, tfs = idx.postings[t]
            dot[docs] += wq * (1.0 + np.log10(tfs)) * self.idf[t]
        q_norm = math.sqrt(q_sq)
        scores = np.zeros_like(dot)
        mask = (dot > 0) & (self.doc_norm > 0)
        if q_norm > 0:
            scores[mask] = dot[mask] / (q_norm * self.doc_norm[mask])
        return scores
