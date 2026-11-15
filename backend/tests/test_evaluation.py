"""Uso: uv run python -m backend.tests.test_evaluation"""
from backend.app.evaluation import evaluate, query_metrics


def main() -> None:
    # ranking [a,b,c,d], relevantes {a,c,e}
    m = query_metrics(["a", "b", "c", "d"], {"a", "c", "e"}, k_values=(2, 4))
    assert m["precision@2"] == 0.5 and abs(m["recall@2"] - 1 / 3) < 1e-12
    assert m["precision@4"] == 0.5 and abs(m["recall@4"] - 2 / 3) < 1e-12
    assert abs(m["ap"] - (1 / 1 + 2 / 3) / 3) < 1e-12  # = 0.5556

    queries = {"q1": "x", "q2": "y"}
    qrels = {"q1": {"a": 1, "c": 1, "e": 1}, "q2": {"z": 1}}
    runs = {"x": [("a", 3), ("b", 2), ("c", 1), ("d", 0.5)], "y": [("z", 1)]}
    r = evaluate(lambda q, k: runs[q], queries, qrels, k_values=(2,), depth=10)
    assert r["n_queries"] == 2
    assert abs(r["summary"]["MAP@10"] - ((1 + 2 / 3) / 3 + 1.0) / 2) < 1e-12
    print("OK métricas:", r["summary"])


if __name__ == "__main__":
    main()
