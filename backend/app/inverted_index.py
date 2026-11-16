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
    # Los postings se guardan en dos arreglos contiguos (formato CSR) y se leen de una sola vez:
    # cargarlos término por término fragmenta la memoria y triplica el uso de RAM (importa en Render).
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        terms = list(self.postings)
        lengths = np.array([len(self.postings[t][0]) for t in terms], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        docs_all = np.concatenate([self.postings[t][0] for t in terms]).astype(np.int32, copy=False)
        tfs_all = np.concatenate([self.postings[t][1] for t in terms]).astype(np.int32, copy=False)
        state = {
            "doc_ids": self.doc_ids, "doc_len": self.doc_len, "doc_nterms": self.doc_nterms,
            "csr": (terms, offsets, docs_all, tfs_all),
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        with gzip.open(path, "rb") as f:
            raw = f.read()
        state = pickle.loads(raw)
        del raw
        idx = cls()
        idx.doc_ids, idx.doc_len, idx.doc_nterms = state["doc_ids"], state["doc_len"], state["doc_nterms"]
        if "csr" in state:
            terms, offsets, docs_all, tfs_all = state["csr"]
            idx.postings = {t: (docs_all[offsets[i]:offsets[i + 1]], tfs_all[offsets[i]:offsets[i + 1]])
                            for i, t in enumerate(terms)}
        else:  # formato anterior (un par de arreglos por término)
            idx.postings = state["postings"]
        return idx
