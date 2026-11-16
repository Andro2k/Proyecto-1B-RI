"""Evalúa los modelos sobre las consultas de TEST y guarda backend/artifacts/metrics.json
(métricas globales, detalle por consulta, latencia y configuración usada).
Uso: uv run python -m backend.scripts.run_eval"""
import json
import time

from backend.app.data import load_qrels, load_queries
from backend.app.evaluation import evaluate
from backend.app.registry import ARTIFACTS, INDEX_PATH, build_models, load_config

METRICS_PATH = ARTIFACTS / "metrics.json"


def main() -> None:
    queries, qrels = load_queries(only_with_qrels="test"), load_qrels("test")
    models = build_models()
    if "embeddings" in models:
        models["embeddings"].search("calentamiento del modelo ONNX", 1)  # la carga inicial no cuenta como latencia

    results = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        results[name] = evaluate(m.search, queries, qrels)
        n = results[name]["n_queries"]
        results[name]["latency_ms"] = (time.perf_counter() - t0) / n * 1000  # media por consulta (incluye calcular métricas)
        print(f"{name}: {n} consultas, {results[name]['latency_ms']:.1f} ms/consulta")

    results["_config"] = {**load_config(), "split": "test", "n_queries": len(queries),
                          "index_mb": round(INDEX_PATH.stat().st_size / 1e6, 1)}
    cols = ["precision@10", "recall@10", "ndcg@10", "recall@20", "MAP@100"]
    print(f"\n{'modelo':<12}" + "".join(f"{c:>14}" for c in cols) + f"{'ms/consulta':>14}")
    for name, r in results.items():
        if name.startswith("_"):
            continue
        print(f"{name:<12}" + "".join(f"{r['summary'][c]:>14.4f}" for c in cols) + f"{r['latency_ms']:>14.1f}")

    METRICS_PATH.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {METRICS_PATH}")


if __name__ == "__main__":
    main()
