import numpy as np

from ..inverted_index import InvertedIndex


class RetrievalModel:
    """Interfaz común: search(query, k) -> [(doc_id, score), ...] ordenado de mayor a menor."""

    name = "base"

    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    def score_all(self, query: str) -> np.ndarray:
        """Vector de puntajes de tamaño n_docs (0 para documentos sin términos en común)."""
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.score_all(query)
        return top_k(scores, self.index.doc_ids, k)


def top_k(scores: np.ndarray, doc_ids: list[str], k: int) -> list[tuple[str, float]]:
    """Top-k de documentos con puntaje > 0 (desempate estable por posición)."""
    k = min(k, int((scores > 0).sum()))
    if k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.lexsort((part, -scores[part]))]
    return [(doc_ids[i], float(scores[i])) for i in order]
