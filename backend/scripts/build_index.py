"""Construye el índice invertido y lo guarda en backend/artifacts/index.pkl.gz
Uso: uv run python -m backend.scripts.build_index"""
import time
from pathlib import Path

from backend.app.data import load_corpus
from backend.app.inverted_index import InvertedIndex

INDEX_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "index.pkl.gz"


def main() -> None:
    t0 = time.time()
    corpus = load_corpus()
    print(f"Corpus cargado: {len(corpus)} documentos")
    index = InvertedIndex.build(corpus)
    index.save(INDEX_PATH)
    print(f"Vocabulario: {len(index.postings)} términos | avgdl: {index.avgdl:.1f}")
    print(f"Guardado en {INDEX_PATH} ({INDEX_PATH.stat().st_size / 1e6:.1f} MB) en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
