"""Elige parámetros en las consultas de DESARROLLO (qrels dev), sin tocar las de prueba, como pide la
clase de Evaluación ("usar consultas de desarrollo para elegir parámetros... reservar prueba").
Guarda backend/artifacts/config.json (mejores valores) y tuning.json (toda la rejilla, para el informe).
Criterio de selección: MAP@100 en dev.   Uso: uv run python -m backend.scripts.tune"""
import itertools
import json

from backend.app.data import load_qrels, load_queries
from backend.app.evaluation import evaluate
from backend.app.inverted_index import InvertedIndex
from backend.app.models import BM25Model, TfidfModel
from backend.app.registry import ARTIFACTS, CONFIG_PATH, INDEX_PATH

K1_GRID = (0.6, 0.9, 1.2, 1.5, 1.8, 2.0)
B_GRID = (0.25, 0.5, 0.75, 0.9, 1.0)
TF_GRID = ("raw", "sublinear")
IDF_GRID = ("ln1", "log10", "smooth")


def score(model, queries, qrels) -> dict:
    s = evaluate(model.search, queries, qrels)["summary"]
    return {"MAP@100": s["MAP@100"], "ndcg@10": s["ndcg@10"], "recall@20": s["recall@20"]}


def main() -> None:
    index = InvertedIndex.load(INDEX_PATH)
    queries, qrels = load_queries(only_with_qrels="dev"), load_qrels("dev")
    print(f"Consultas de desarrollo: {len(queries)}")

    bm25 = [{"k1": k1, "b": b, **score(BM25Model(index, k1=k1, b=b), queries, qrels)}
            for k1, b in itertools.product(K1_GRID, B_GRID)]
    tfidf = [{"tf": tf, "idf": idf, **score(TfidfModel(index, tf=tf, idf=idf), queries, qrels)}
             for tf, idf in itertools.product(TF_GRID, IDF_GRID)]

    for title, rows, keys in (("BM25", bm25, ("k1", "b")), ("TF-IDF", tfidf, ("tf", "idf"))):
        print(f"\n{title} (ordenado por MAP@100 en dev)")
        for r in sorted(rows, key=lambda r: -r["MAP@100"]):
            print("  " + "  ".join(f"{k}={r[k]}" for k in keys) + f"   MAP@100={r['MAP@100']:.4f}  nDCG@10={r['ndcg@10']:.4f}")

    best_bm25 = max(bm25, key=lambda r: r["MAP@100"])
    best_tfidf = max(tfidf, key=lambda r: r["MAP@100"])
    base = next(r for r in bm25 if r["k1"] == 1.2 and r["b"] == 0.75)
    print(f"\nBM25 de clase (k1=1.2, b=0.75): MAP@100={base['MAP@100']:.4f}  | mejor: k1={best_bm25['k1']}, b={best_bm25['b']} -> {best_bm25['MAP@100']:.4f}")

    config = {"bm25": {"k1": best_bm25["k1"], "b": best_bm25["b"]},
              "tfidf": {"tf": best_tfidf["tf"], "idf": best_tfidf["idf"]}}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (ARTIFACTS / "tuning.json").write_text(
        json.dumps({"n_dev_queries": len(queries), "bm25": bm25, "tfidf": tfidf}, indent=2), encoding="utf-8")
    print(f"Guardado {CONFIG_PATH.name} y tuning.json en {ARTIFACTS}")


if __name__ == "__main__":
    main()
