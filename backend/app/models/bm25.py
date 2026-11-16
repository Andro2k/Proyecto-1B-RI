import math

import numpy as np

from ..inverted_index import InvertedIndex
from ..preprocessing import preprocess
from .base import RetrievalModel


class BM25Model(RetrievalModel):
    """BM25 (Okapi): idf = ln(1 + (N - df + 0.5) / (df + 0.5)),
    score = Σ idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))."""

    name = "bm25"

    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75, analyzer=preprocess) -> None:
        super().__init__(index, analyzer)
        self.k1, self.b = k1, b
        self.norm = k1 * (1.0 - b + b * index.doc_len / index.avgdl)  # factor por documento

    def idf(self, term: str) -> float:
        n, df = self.index.n_docs, self.index.df(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_all(self, query: str) -> np.ndarray:
        idx = self.index
        scores = np.zeros(idx.n_docs, dtype=np.float64)
        for t in set(self.analyzer(query)):
            if t not in idx.postings:
                continue
            docs, tfs = idx.postings[t]
            scores[docs] += self.idf(t) * tfs * (self.k1 + 1.0) / (tfs + self.norm[docs])
        return scores
