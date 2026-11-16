"""Compara dos modelos consulta por consulta (punto f del proyecto): dónde gana y dónde pierde uno
frente al otro. Un promedio puede ocultar diferencias, así que se reportan: victorias/derrotas/empates,
IC 95% bootstrap pareado de la diferencia media, resultados por tipo de consulta y los casos extremos.
Lee backend/artifacts/metrics.json (generado por run_eval).
Uso: uv run python -m backend.scripts.compare_models [--a embeddings] [--b bm25] [--metric ap]"""
import argparse
import json
import re

import numpy as np

from backend.app.data import load_queries
from backend.app.registry import ARTIFACTS

BUCKETS = {
    "consulta corta (<= 6 palabras)": lambda q: len(q.split()) <= 6,
    "consulta media (7-12 palabras)": lambda q: 7 <= len(q.split()) <= 12,
    "consulta larga (> 12 palabras)": lambda q: len(q.split()) > 12,
    "contiene cifras": lambda q: bool(re.search(r"\d", q)),
    "contiene siglas/tickers (MAYÚSCULAS)": lambda q: bool(re.search(r"\b[A-Z]{2,}\b", q)),
}


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(per_a: dict, per_b: dict, queries: dict, metric: str, top: int = 5) -> dict:
    qids = sorted(set(per_a) & set(per_b))
    a = np.array([per_a[q][metric] for q in qids])
    b = np.array([per_b[q][metric] for q in qids])
    diff = a - b
    lo, hi = paired_bootstrap_ci(diff)
    out = {
        "n": len(qids), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "a_gana": int((diff > 1e-9).sum()), "b_gana": int((diff < -1e-9).sum()), "empates": int((abs(diff) <= 1e-9).sum()),
        "diff_media": float(diff.mean()), "ic95": [lo, hi],
        "buckets": {}, "a_mejora_mas": [], "a_empeora_mas": [],
    }
    for name, cond in BUCKETS.items():
        idx = [i for i, q in enumerate(qids) if cond(queries.get(q, ""))]
        if idx:
            out["buckets"][name] = {"n": len(idx), "a": float(a[idx].mean()), "b": float(b[idx].mean())}
    order = np.argsort(diff)
    row = lambda i: {"qid": qids[i], "query": queries.get(qids[i], ""), "a": float(a[i]), "b": float(b[i])}
    out["a_empeora_mas"] = [row(i) for i in order[:top]]
    out["a_mejora_mas"] = [row(i) for i in order[::-1][:top]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="embeddings")
    ap.add_argument("--b", default="bm25")
    ap.add_argument("--metric", default="ap", help="ap, ndcg@10, recall@20, precision@10, ...")
    args = ap.parse_args()

    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    for m in (args.a, args.b):
        if m not in metrics:
            raise SystemExit(f"'{m}' no está en metrics.json (modelos: {[k for k in metrics if not k.startswith('_')]})")
    res = compare(metrics[args.a]["per_query"], metrics[args.b]["per_query"], load_queries(), args.metric)

    A, B = args.a, args.b
    print(f"{A} vs {B} | métrica: {args.metric} | {res['n']} consultas")
    print(f"  media {A}: {res['mean_a']:.4f}   media {B}: {res['mean_b']:.4f}   diferencia: {res['diff_media']:+.4f}  IC95% [{res['ic95'][0]:+.4f}, {res['ic95'][1]:+.4f}]")
    print(f"  {A} gana en {res['a_gana']} consultas, {B} gana en {res['b_gana']}, empatan {res['empates']}")
    print("\nPor tipo de consulta:")
    for name, r in res["buckets"].items():
        print(f"  {name:<40} n={r['n']:<4} {A}={r['a']:.4f}  {B}={r['b']:.4f}  ({r['a'] - r['b']:+.4f})")
    for title, key in ((f"{A} mejora más sobre {B}", "a_mejora_mas"), (f"{A} empeora más respecto a {B}", "a_empeora_mas")):
        print(f"\n{title}:")
        for r in res[key]:
            print(f"  [{r['qid']}] {A}={r['a']:.3f} {B}={r['b']:.3f}  {r['query'][:110]}")

    out = ARTIFACTS / f"comparison_{A}_vs_{B}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
