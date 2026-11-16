from typing import Callable

import numpy as np

from ..inverted_index import InvertedIndex
from ..preprocessing import preprocess


class RetrievalModel:
    """Interfaz común: search(query, k) -> [(doc_id, score), ...] ordenado de mayor a menor.
    Se devuelven solo documentos con score > 0; el empate se resuelve por doc_id (cadena) ascendente.
    El `analyzer` (consulta -> términos) debe ser el mismo con que se construyó el índice."""

    name = "base"

    def __init__(self, index: InvertedIndex, analyzer: Callable[[str], list[str]] = preprocess) -> None:
        self.index = index
        self.analyzer = analyzer

    def score_all(self, query: str) -> np.ndarray:
        """Vector de puntajes de tamaño n_docs (0 para documentos sin términos en común)."""
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return top_k(self.score_all(query), self.index.doc_ids, self.index.id_rank, k)


def top_k(scores: np.ndarray, doc_ids: list[str], id_rank: np.ndarray, k: int) -> list[tuple[str, float]]:
    """Top-k de documentos con puntaje > 0. Incluye todos los empatados en el corte antes de
    desempatar por id, para que el resultado no dependa del orden interno del corpus."""
    pos = np.flatnonzero(scores > 0)
    if k <= 0 or pos.size == 0:
        return []
    if pos.size > k:
        kth = np.partition(scores[pos], pos.size - k)[pos.size - k]  # k-ésimo mayor valor
        pos = pos[scores[pos] >= kth]
    order = pos[np.lexsort((id_rank[pos], -scores[pos]))][:k]
    return [(doc_ids[i], float(scores[i])) for i in order]
