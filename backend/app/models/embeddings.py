from pathlib import Path

from ..vector_store import load_store


class EmbeddingModel:
    """Recuperación semántica: embedding de la consulta + búsqueda vectorial en FAISS."""

    name = "embeddings"

    def __init__(self, store_dir: Path, embedder=None) -> None:
        self.index, self.meta = load_store(store_dir)
        self.doc_ids: list[str] = self.meta["doc_ids"]
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:  # el modelo ONNX se carga la primera vez que se consulta
            from ..embedder import FastEmbedder

            self._embedder = FastEmbedder(self.meta["model_name"], self.meta["max_length"])
        return self._embedder

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        vec = self.embedder.embed_query(query)
        scores, ids = self.index.search(vec, k)
        return [(self.doc_ids[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
