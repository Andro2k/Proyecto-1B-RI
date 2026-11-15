"""Prueba el camino FAISS (construcción, guardado, carga y búsqueda) con un embedder falso.
Uso: uv run python -m backend.tests.test_vector_store"""
import tempfile
import zlib
from pathlib import Path

import numpy as np

from backend.app.models.embeddings import EmbeddingModel
from backend.app.vector_store import build_faiss, save_store


class FakeEmbedder:
    """Embedding determinista por bolsa de palabras hasheadas (solo para probar la tubería)."""

    dim = 4096

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().split():
            v[zlib.crc32(w.encode()) % self.dim] += 1
        return v / max(np.linalg.norm(v), 1e-12)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)[None, :]


def main() -> None:
    docs = {f"d{i}": f"palabra{i} comun otra{i % 7}" for i in range(500)}
    emb = FakeEmbedder()
    vecs = emb.embed_documents(list(docs.values()))
    with tempfile.TemporaryDirectory() as tmp:
        for fp16 in (False, True):
            save_store(build_faiss(vecs, fp16=fp16), list(docs), "fake", 256, Path(tmp))
            model = EmbeddingModel(Path(tmp), embedder=emb)
            top = model.search("palabra42 comun", 5)
            assert top[0][0] == "d42", top
            assert all(top[i][1] >= top[i + 1][1] for i in range(4))
            print(f"OK fp16={fp16}:", top[:2])
    print("Capa vectorial correcta.")


if __name__ == "__main__":
    main()
