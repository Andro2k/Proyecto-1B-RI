"""Métricas de evaluación (convenciones declaradas, como pide la clase de Evaluación).

Independientes del modelo: reciben search(query, depth) -> [(doc_id, score), ...] de mayor a menor.

Relevancia: los qrels dan un grado entero por par (consulta, documento). Es relevante si rel >= 1;
los documentos sin juicio se tratan como no relevantes (los qrels de FiQA son incompletos).
Por consulta:
  * Precision@k = aciertos en el top-k / k. Si el sistema devuelve menos de k documentos, las
    posiciones faltantes cuentan como no relevantes (el denominador sigue siendo k).
  * Recall@k    = aciertos en el top-k / |relevantes de la consulta|.
  * NDCG@k      = DCG@k / IDCG@k, con ganancia 2^rel - 1 y descuento 1/log2(i + 1); el IDCG se
    construye con TODOS los documentos juzgados de la consulta (no solo los recuperados).
  * AP          = promedio de la precisión en cada posición con acierto, sobre los primeros
    `depth` resultados y dividido entre el total de relevantes de la consulta.
Sistema: promedio simple sobre consultas (MAP@depth es el promedio de AP). Las consultas sin
ningún relevante se excluyen del promedio (IDCG = 0 y Recall indefinido)."""
import math
from typing import Callable

K_VALUES = (5, 10, 20)
DEPTH = 100

SearchFn = Callable[[str, int], list[tuple[str, float]]]


def _dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(grades, 1))


def query_metrics(ranked_ids: list[str], rels: dict[str, int], k_values=K_VALUES) -> dict[str, float]:
    """`rels`: doc_id -> grado de relevancia (solo grados >= 1)."""
    out: dict[str, float] = {}
    for k in k_values:
        hits = sum(1 for d in ranked_ids[:k] if d in rels)
        out[f"precision@{k}"] = hits / k
        out[f"recall@{k}"] = hits / len(rels)
        ideal = sorted(rels.values(), reverse=True)[:k]
        out[f"ndcg@{k}"] = _dcg([rels.get(d, 0) for d in ranked_ids[:k]]) / _dcg(ideal)
    hits, ap = 0, 0.0
    for rank, d in enumerate(ranked_ids, 1):
        if d in rels:
            hits += 1
            ap += hits / rank
    out["ap"] = ap / len(rels)
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
        rels = {d: g for d, g in qrels.get(qid, {}).items() if g >= 1}
        if not rels:
            continue
        ranked = [d for d, _ in search(text, max(depth, max(k_values)))]
        per_query[qid] = query_metrics(ranked, rels, k_values)
    n = len(per_query)
    keys = list(next(iter(per_query.values()))) if n else []
    mean = {k: sum(m[k] for m in per_query.values()) / n for k in keys}
    summary = {k: v for k, v in mean.items() if k != "ap"}
    summary[f"MAP@{depth}"] = mean.get("ap", 0.0)
    return {"n_queries": n, "summary": summary, "per_query": per_query}
