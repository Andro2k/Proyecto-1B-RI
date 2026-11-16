"""Verifica las métricas con el ejemplo de la clase de Evaluación y con casos a mano.
Uso: uv run python -m backend.tests.test_evaluation"""
from backend.app.evaluation import evaluate, query_metrics

# Ejemplo de la clase: R = {d1: 3, d2: 2, d3: 2, d4: 1}
RELS = {"d1": 3, "d2": 2, "d3": 2, "d4": 1}
A = ["d5", "d2", "d1", "d6", "d3", "d4", "d7", "d8"]
B = ["d1", "d2", "d5", "d3", "d6", "d4", "d7", "d8"]


def main() -> None:
    a, b = query_metrics(A, RELS, (5,)), query_metrics(B, RELS, (5,))
    for m in (a, b):  # mismos documentos en el top-5: P@5 = 3/5 y R@5 = 3/4
        assert abs(m["precision@5"] - 0.60) < 1e-12 and abs(m["recall@5"] - 0.75) < 1e-12
    assert abs(a["ndcg@5"] - 0.605) < 5e-4 and abs(b["ndcg@5"] - 0.941) < 5e-4  # diapositivas 21-23

    # ranking [a,b,c,d], relevantes {a,c,e}, todos de grado 1
    m = query_metrics(["a", "b", "c", "d"], {"a": 1, "c": 1, "e": 1}, k_values=(2, 4))
    assert m["precision@2"] == 0.5 and abs(m["recall@2"] - 1 / 3) < 1e-12
    assert m["precision@4"] == 0.5 and abs(m["recall@4"] - 2 / 3) < 1e-12
    assert abs(m["ap"] - (1 / 1 + 2 / 3) / 3) < 1e-12  # 0.5556

    # Un sistema que devuelve menos de k documentos: las posiciones faltantes cuentan como no relevantes
    assert query_metrics(["a"], {"a": 1}, (5,))["precision@5"] == 0.2

    queries = {"q1": "x", "q2": "y", "q3": "z"}
    qrels = {"q1": {"a": 1, "c": 1, "e": 1}, "q2": {"z": 1}}  # q3 sin juicios: se excluye
    runs = {"x": [("a", 3), ("b", 2), ("c", 1), ("d", 0.5)], "y": [("z", 1)], "z": []}
    r = evaluate(lambda q, k: runs[q], queries, qrels, k_values=(2,), depth=10)
    assert r["n_queries"] == 2
    assert abs(r["summary"]["MAP@10"] - ((1 + 2 / 3) / 3 + 1.0) / 2) < 1e-12
    print(f"OK métricas (ejemplo de clase: NDCG@5 A={a['ndcg@5']:.3f}, B={b['ndcg@5']:.3f})")


if __name__ == "__main__":
    main()
