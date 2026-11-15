"""Prueba rápida de los 3 modelos clásicos. Marca con [*] los documentos relevantes según los qrels de test.
Uso: uv run python -m backend.scripts.try_models ["consulta libre"]"""
import sys

from backend.app.data import load_corpus, load_qrels, load_queries
from backend.app.inverted_index import InvertedIndex
from backend.app.models import BM25Model, JaccardModel, TfidfModel
from backend.app.models.embeddings import EmbeddingModel
from backend.scripts.build_index import INDEX_PATH

STORE_DIR = INDEX_PATH.parent


def main() -> None:
    index = InvertedIndex.load(INDEX_PATH)
    corpus = load_corpus()
    qrels = load_qrels("test")
    if len(sys.argv) > 1:
        query, rel = " ".join(sys.argv[1:]), {}
    else:
        qid, query = next(iter(load_queries().items()))
        rel = qrels[qid]
    print(f"Consulta: {query}\n")
    models = [JaccardModel(index), TfidfModel(index), BM25Model(index)]
    if (STORE_DIR / "faiss.index").exists():
        models.append(EmbeddingModel(STORE_DIR))
    for model in models:
        print(f"== {model.name}")
        for rank, (did, score) in enumerate(model.search(query, 5), 1):
            mark = "[*]" if did in rel else "   "
            print(f"{rank}. {mark} {score:.4f}  {did}: {corpus[did]['text'][:90]!r}")
        print()


if __name__ == "__main__":
    main()
