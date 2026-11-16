"""Reproduce los ejemplos numéricos de las diapositivas de la asignatura (Prof. Carrera).
Uso: uv run python -m backend.tests.test_class_examples"""
import re

from backend.app.inverted_index import InvertedIndex
from backend.app.models import BM25Model, JaccardModel, TfidfModel

ES_STOP = {"el", "la", "en", "con", "y"}


def es_analyzer(text: str) -> list[str]:
    """Preprocesamiento de la clase: minúsculas, sin puntuación, sin stopwords; conserva tildes."""
    return [t for t in re.findall(r"\w+", text.lower()) if t not in ES_STOP]


def close(a, b, tol=5e-5):
    assert abs(a - b) < tol, (a, b)


def bm25_example() -> None:
    """Diapositivas BM25: consulta «datos ciencia», k1=1.2, b=0.75."""
    corpus = {
        "D1": {"title": "", "text": "datos datos ciencia"},
        "D2": {"title": "", "text": "datos ciencia modelos aprendizaje"},
        "D3": {"title": "", "text": "ciencia modelos"},
    }
    idx = InvertedIndex.build(corpus, progress_every=0, analyzer=es_analyzer)
    assert idx.avgdl == 3.0
    bm25 = BM25Model(idx, k1=1.2, b=0.75, analyzer=es_analyzer)
    s = dict(bm25.search("datos ciencia", 10))
    close(s["D1"], 0.7798); close(s["D2"], 0.5311); close(s["D3"], 0.1546)
    # Misma colección con TF-IDF + coseno, TF cruda e IDF suavizada: 0.9583 / 0.6083 / 0.3762
    tf = dict(TfidfModel(idx, tf="raw", idf="smooth", analyzer=es_analyzer).search("datos ciencia", 10))
    close(tf["D1"], 0.9583); close(tf["D2"], 0.6083); close(tf["D3"], 0.3762)
    print("OK BM25 y TF-IDF (idf suavizada) = diapositivas BM25")


def vector_example() -> None:
    """Diapositivas «Modelo vectorial»: consulta «gato juega», IDF = log10(N/df), TF cruda."""
    docs = [
        "El perro duerme en la casa.", "El gato duerme en la casa.", "El perro juega con el gato.",
        "El gato juega y juega en el jardín.", "El pájaro canta en el jardín.",
    ]
    corpus = {f"D{i + 1}": {"title": "", "text": t} for i, t in enumerate(docs)}
    idx = InvertedIndex.build(corpus, progress_every=0, analyzer=es_analyzer)
    tfidf = TfidfModel(idx, tf="raw", idf="log10", analyzer=es_analyzer)
    top = tfidf.search("gato juega", 10)
    assert [d for d, _ in top] == ["D4", "D3", "D2"], top  # D1 y D5 quedan en cero y no se devuelven
    close(top[0][1], 0.8758); close(top[1][1], 0.7532); close(top[2][1], 0.1786)
    # Jaccard binario: D3 = {perro, juega, gato} y D4 = {gato, juega, jardín} dan 2/3; D2 = {gato, duerme, casa}, 1/4.
    # D3 y D4 empatan: se desempata por doc_id ascendente.
    jac = JaccardModel(idx, analyzer=es_analyzer).search("gato juega", 10)
    assert [d for d, _ in jac] == ["D3", "D4", "D2"], jac
    close(jac[0][1], 2 / 3); close(jac[1][1], 2 / 3); close(jac[2][1], 1 / 4)
    print("OK TF-IDF (log10) = diapositivas del modelo vectorial:", [(d, round(s, 4)) for d, s in top])


if __name__ == "__main__":
    bm25_example()
    vector_example()
    print("Los ejemplos de clase se reproducen.")
