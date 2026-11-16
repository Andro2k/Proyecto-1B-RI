"""Índice invertido: término -> (doc_idx[], tf[]), más estadísticas de cada documento."""
import gzip
import pickle
from functools import cached_property
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from .preprocessing import preprocess


class InvertedIndex:
    def __init__(self) -> None:
        self.doc_ids: list[str] = []          # posición -> id original del documento
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.int32)    # nº de términos (con repeticiones)
        self.doc_nterms: np.ndarray = np.zeros(0, dtype=np.int32)  # nº de términos distintos
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ---- construcción -------------------------------------------------
    @classmethod
    def build(
        cls,
        corpus: dict[str, dict[str, str]],
        progress_every: int = 10000,
        analyzer: Callable[[str], list[str]] = preprocess,
    ) -> "InvertedIndex":
        idx = cls()
        tmp: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        lens, nterms = [], []
        for i, (doc_id, doc) in enumerate(corpus.items()):
            idx.doc_ids.append(doc_id)
            counts = Counter(analyzer(f"{doc['title']} {doc['text']}"))
            lens.append(sum(counts.values()))
            nterms.append(len(counts))
            for term, tf in counts.items():
                docs, tfs = tmp[term]
                docs.append(i)
                tfs.append(tf)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  indexados {i + 1} documentos...")
        idx.doc_len = np.array(lens, dtype=np.int32)
        idx.doc_nterms = np.array(nterms, dtype=np.int32)
        idx.postings = {
            t: (np.array(d, dtype=np.int32), np.array(f, dtype=np.int32)) for t, (d, f) in tmp.items()
        }
        return idx

    # ---- estadísticas -------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def avgdl(self) -> float:
        return float(self.doc_len.mean())

    @cached_property
    def id_rank(self) -> np.ndarray:
        """Posición de cada documento al ordenar sus IDs como cadenas (desempate determinista)."""
        order = np.argsort(np.array(self.doc_ids, dtype=object), kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        return rank

    def df(self, term: str) -> int:
        return len(self.postings[term][0]) if term in self.postings else 0

    # ---- persistencia -------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as f:
            state = {k: v for k, v in self.__dict__.items() if k != "id_rank"}
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        idx = cls()
        with gzip.open(path, "rb") as f:
            idx.__dict__.update(pickle.load(f))
        return idx
