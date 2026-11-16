import math
from collections import Counter

import numpy as np

from ..inverted_index import InvertedIndex
from ..preprocessing import preprocess
from .base import RetrievalModel

# Variantes de IDF usadas en la asignatura (N = nº de documentos, df = frecuencia documental)
IDF_VARIANTS = {
    "ln1": lambda n, df: math.log(n / df) + 1.0,                 # notebook 03 (= sklearn smooth_idf=False)
    "log10": lambda n, df: math.log10(n / df),                    # diapositivas "Modelo vectorial"
    "smooth": lambda n, df: 1.0 + math.log((n + 1) / (df + 1)),   # diapositivas BM25 (= sklearn smooth_idf=True)
}


def tf_weight(tf, kind: str):
    """Peso local: "raw" = f ; "sublinear" = 1 + ln f (f > 0)."""
    return (1.0 + np.log(tf)) if kind == "sublinear" else tf


class TfidfModel(RetrievalModel):
    """Coseno entre vectores w = TF * IDF de documento y consulta (la consulta usa el vocabulario
    y los IDF del corpus). Por defecto, la convención del notebook 03: TF cruda e IDF = ln(N/df) + 1."""

    name = "tfidf"

    def __init__(self, index: InvertedIndex, tf: str = "raw", idf: str = "ln1", analyzer=preprocess) -> None:
        super().__init__(index, analyzer)
        self.tf_kind, self.idf_kind = tf, idf
        n, idf_fn = index.n_docs, IDF_VARIANTS[idf]
        self.idf = {t: idf_fn(n, len(p[0])) for t, p in index.postings.items()}
        # Norma euclídea de cada documento (se calcula una sola vez recorriendo los postings)
        sq = np.zeros(n, dtype=np.float64)
        for t, (docs, tfs) in index.postings.items():
            w = tf_weight(tfs.astype(np.float64), tf) * self.idf[t]
            sq[docs] += w * w
        self.doc_norm = np.sqrt(sq)

    def score_all(self, query: str) -> np.ndarray:
        idx = self.index
        q_tf = Counter(t for t in self.analyzer(query) if t in idx.postings)
        dot = np.zeros(idx.n_docs, dtype=np.float64)
        q_sq = 0.0
        for t, tf in q_tf.items():
            wq = float(tf_weight(float(tf), self.tf_kind)) * self.idf[t]
            q_sq += wq * wq
            docs, tfs = idx.postings[t]
            dot[docs] += wq * tf_weight(tfs.astype(np.float64), self.tf_kind) * self.idf[t]
        q_norm = math.sqrt(q_sq)
        scores = np.zeros_like(dot)
        mask = (dot > 0) & (self.doc_norm > 0)
        if q_norm > 0:
            scores[mask] = dot[mask] / (q_norm * self.doc_norm[mask])
        return scores
