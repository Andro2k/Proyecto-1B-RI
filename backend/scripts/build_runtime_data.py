"""Genera los datos que necesita la web en producción (el corpus completo no se sube al repositorio):
  backend/artifacts/docs.jsonl.gz   textos de los documentos (se truncan a MAX_CHARS para mostrarlos)
  backend/artifacts/benchmark.json  consultas de prueba y qrels de test (para marcar documentos relevantes)
Uso: uv run python -m backend.scripts.build_runtime_data"""
import gzip
import json

from backend.app.data import load_corpus, load_qrels, load_queries
from backend.app.registry import ARTIFACTS

MAX_CHARS = 2000


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    truncated = 0
    with gzip.open(ARTIFACTS / "docs.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as f:
        for doc_id, d in corpus.items():
            text = d["text"]
            if len(text) > MAX_CHARS:
                text, truncated = text[:MAX_CHARS].rstrip() + "…", truncated + 1
            f.write(json.dumps({"id": doc_id, "title": d["title"], "text": text}, ensure_ascii=False) + "\n")
    size = (ARTIFACTS / "docs.jsonl.gz").stat().st_size / 1e6
    print(f"docs.jsonl.gz: {len(corpus)} documentos, {truncated} truncados a {MAX_CHARS} caracteres, {size:.1f} MB")

    queries, qrels = load_queries(only_with_qrels="test"), load_qrels("test")
    bench = {"split": "test", "queries": queries, "qrels": {q: qrels[q] for q in queries}}
    (ARTIFACTS / "benchmark.json").write_text(json.dumps(bench, ensure_ascii=False), encoding="utf-8")
    print(f"benchmark.json: {len(queries)} consultas de prueba")


if __name__ == "__main__":
    main()
