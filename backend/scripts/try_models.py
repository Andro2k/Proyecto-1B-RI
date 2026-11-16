"""Prueba rápida de los modelos. Marca con [*] los documentos relevantes según los qrels de test.
Uso: uv run python -m backend.scripts.try_models ["consulta libre"]"""
import sys

from backend.app.data import load_corpus, load_qrels, load_queries
from backend.app.registry import build_models


def main() -> None:
    corpus = load_corpus()
    if len(sys.argv) > 1:
        query, rel = " ".join(sys.argv[1:]), {}
    else:
        qid, query = next(iter(load_queries().items()))
        rel = load_qrels("test")[qid]
    print(f"Consulta: {query}\n")
    for name, model in build_models().items():
        print(f"== {name}")
        for rank, (did, score) in enumerate(model.search(query, 5), 1):
            mark = "[*]" if did in rel else "   "
            print(f"{rank}. {mark} {score:.4f}  {did}: {corpus[did]['text'][:90]!r}")
        print()


if __name__ == "__main__":
    main()
