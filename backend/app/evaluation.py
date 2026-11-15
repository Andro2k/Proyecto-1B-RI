"""Métricas de evaluación. Son independientes del modelo: reciben una función
search(query, depth) -> [(doc_id, score), ...] ordenada de mayor a menor puntaje.

Por consulta:  Precision@k, Recall@k (para cada k en K_VALUES) y AP (average precision sobre los
primeros `depth` documentos, dividida por el total de relevantes de la consulta).
Para todo el sistema: promedio de las anteriores; el promedio de AP es el MAP@depth.
Se toma como relevante todo documento con score > 0 en los qrels."""
from typing import Callable

K_VALUES = (5, 10, 20)
DEPTH = 100

SearchFn = Callable[[str, int], list[tuple[str, float]]]


def query_metrics(ranked_ids: list[str], relevant: set[str], k_values=K_VALUES) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in k_values:
        hits = sum(1 for d in ranked_ids[:k] if d in relevant)
        out[f"precision@{k}"] = hits / k
        out[f"recall@{k}"] = hits / len(relevant)
    hits, ap = 0, 0.0
    for rank, d in enumerate(ranked_ids, 1):
        if d in relevant:
            hits += 1
            ap += hits / rank
    out["ap"] = ap / len(relevant)
    return out


def evaluate(
    search: SearchFn,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k_values=K_VALUES,
    depth: int = DEPTH,
) -> dict:
    per_query: dict[str, dict[str, float]] = {}
    for qid, text in queries.items():
        relevant = {d for d, s in qrels.get(qid, {}).items() if s > 0}
        if not relevant:
            continue
        ranked = [d for d, _ in search(text, max(depth, max(k_values)))]
        per_query[qid] = query_metrics(ranked, relevant, k_values)
    n = len(per_query)
    keys = next(iter(per_query.values())).keys() if n else []
    mean = {k: sum(m[k] for m in per_query.values()) / n for k in keys}
    summary = {k: v for k, v in mean.items() if k != "ap"}
    summary[f"MAP@{depth}"] = mean.get("ap", 0.0)
    return {"n_queries": n, "summary": summary, "per_query": per_query}
