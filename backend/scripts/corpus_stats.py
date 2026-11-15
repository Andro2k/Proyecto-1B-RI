"""Estadísticas del corpus para el informe.  Uso: uv run python -m backend.scripts.corpus_stats"""
import numpy as np

from backend.app.data import load_corpus, load_qrels, load_queries


def main() -> None:
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels("test")

    lens = np.array([len((d["title"] + " " + d["text"]).split()) for d in corpus.values()])
    qlens = np.array([len(t.split()) for t in queries.values()])
    rel_per_q = np.array([len(v) for v in qrels.values()])

    print(f"Documentos: {len(corpus)}")
    print(f"Longitud (palabras): media={lens.mean():.1f} mediana={np.median(lens):.0f} p95={np.percentile(lens, 95):.0f} max={lens.max()}")
    print(f"Consultas de prueba (con qrels test): {len(queries)} | longitud media={qlens.mean():.1f}")
    print(f"Relevantes por consulta: media={rel_per_q.mean():.2f} min={rel_per_q.min()} max={rel_per_q.max()}")
    qid = next(iter(queries))
    print(f"\nEjemplo -> [{qid}] {queries[qid]}")
    for did in qrels[qid]:
        print(f"  relevante {did}: {corpus[did]['text'][:200]}...")


if __name__ == "__main__":
    main()
