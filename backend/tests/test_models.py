"""Verifica los modelos contra implementaciones ingenuas (sin índice) en un corpus de juguete.
Uso: uv run python -m backend.tests.test_models"""
import math
from collections import Counter

from backend.app.inverted_index import InvertedIndex
from backend.app.models import BM25Model, JaccardModel, TfidfModel
from backend.app.preprocessing import preprocess

CORPUS = {
    "d1": {"title": "Emergency fund", "text": "Keep your emergency fund in a savings account at a bank."},
    "d2": {"title": "Stocks", "text": "Stocks are shares of a company traded on the stock market."},
    "d3": {"title": "Bonds", "text": "Bonds pay interest. A bond is a loan to a company or government."},
    "d4": {"title": "", "text": "Savings accounts pay low interest but your savings are safe in a bank."},
    "d5": {"title": "Taxes", "text": "Capital gains taxes apply when you sell stocks at a profit."},
}
QUERY = "where to keep emergency savings in a bank"


def naive(index_docs, q_terms):
    n = len(index_docs)
    docs = {d: Counter(preprocess(f"{v['title']} {v['text']}")) for d, v in index_docs.items()}
    df = Counter(t for c in docs.values() for t in c)
    avgdl = sum(sum(c.values()) for c in docs.values()) / n
    out = {"jaccard": {}, "tfidf": {}, "bm25": {}}
    qc = Counter(q_terms)
    for d, c in docs.items():
        inter = set(c) & set(qc)
        out["jaccard"][d] = len(inter) / len(set(c) | set(qc))
        idf = lambda t: math.log(n / df[t]) + 1.0
        w = lambda tf, t: tf * idf(t)
        dn = math.sqrt(sum(w(tf, t) ** 2 for t, tf in c.items()))
        qn = math.sqrt(sum(w(tf, t) ** 2 for t, tf in qc.items() if t in df))
        dot = sum(w(qc[t], t) * w(c[t], t) for t in inter)
        out["tfidf"][d] = dot / (qn * dn) if dot > 0 else 0.0
        dl = sum(c.values())
        s = 0.0
        for t in set(qc):
            if t in c:
                bidf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += bidf * c[t] * 2.2 / (c[t] + 1.2 * (0.25 + 0.75 * dl / avgdl))
        out["bm25"][d] = s
    return out


def main() -> None:
    index = InvertedIndex.build(CORPUS, progress_every=0)
    expected = naive(CORPUS, preprocess(QUERY))
    for model in (JaccardModel(index), TfidfModel(index), BM25Model(index)):
        scores = model.score_all(QUERY)
        for i, d in enumerate(index.doc_ids):
            assert abs(scores[i] - expected[model.name][d]) < 1e-9, (model.name, d, scores[i], expected[model.name][d])
        print(f"OK {model.name:8s} ->", model.search(QUERY, 3))
    print("Todos los modelos coinciden con la implementación ingenua.")


if __name__ == "__main__":
    main()
