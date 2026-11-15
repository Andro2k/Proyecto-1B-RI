"""Evalúa los modelos sobre las consultas de test y guarda backend/artifacts/metrics.json
Uso: uv run python -m backend.scripts.run_eval"""
import json
import time
from pathlib import Path

from backend.app.data import load_qrels, load_queries
from backend.app.evaluation import evaluate
from backend.app.inverted_index import InvertedIndex
from backend.app.models import BM25Model, JaccardModel, TfidfModel
from backend.app.models.embeddings import EmbeddingModel
from backend.scripts.build_index import INDEX_PATH

METRICS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "metrics.json"


def main() -> None:
    index = InvertedIndex.load(INDEX_PATH)
    queries, qrels = load_queries(), load_qrels("test")
    models = [JaccardModel(index), TfidfModel(index), BM25Model(index)]
    if (METRICS_PATH.parent / "faiss.index").exists():
        models.append(EmbeddingModel(METRICS_PATH.parent))
    results = {}
    for m in models:
        t0 = time.time()
        results[m.name] = evaluate(m.search, queries, qrels)
        print(f"{m.name}: evaluado en {time.time() - t0:.1f}s")

    cols = list(next(iter(results.values()))["summary"])
    print(f"\n{'modelo':<10}" + "".join(f"{c:>14}" for c in cols))
    for name, r in results.items():
        print(f"{name:<10}" + "".join(f"{r['summary'][c]:>14.4f}" for c in cols))

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {METRICS_PATH}")


if __name__ == "__main__":
    main()
