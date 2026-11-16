"""Prueba la API con artefactos de juguete (sin corpus real ni modelo ONNX).
Uso: uv run python -m backend.tests.test_api   (requiere httpx, ya incluido en requirements)"""
import gzip
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.inverted_index import InvertedIndex
from backend.app.main import create_app
from backend.app.vector_store import build_faiss, save_store
from backend.tests.test_models import CORPUS
from backend.tests.test_vector_store import FakeEmbedder


def make_artifacts(tmp: Path) -> Path:
    InvertedIndex.build(CORPUS, progress_every=0).save(tmp / "index.pkl.gz")
    emb = FakeEmbedder()
    ids = list(CORPUS)
    save_store(build_faiss(emb.embed_documents([f"{d['title']} {d['text']}" for d in CORPUS.values()])), ids, "fake", 256, tmp)
    with gzip.open(tmp / "docs.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, d in CORPUS.items():
            f.write(json.dumps({"id": i, **d}) + "\n")
    q = "where to keep emergency savings in a bank"
    (tmp / "benchmark.json").write_text(json.dumps({"split": "test", "queries": {"q1": q}, "qrels": {"q1": {"d1": 1, "d4": 1}}}))
    summary = {"precision@10": 0.1, "recall@10": 0.2, "ndcg@10": 0.3, "recall@20": 0.4, "MAP@100": 0.5}
    metrics = {m: {"n_queries": 1, "summary": summary, "latency_ms": 1.0, "per_query": {}} for m in ("jaccard", "tfidf", "bm25", "embeddings")}
    metrics["_config"] = {"bm25": {"k1": 1.2, "b": 0.75}, "n_queries": 1}
    (tmp / "metrics.json").write_text(json.dumps(metrics))
    return tmp


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        art = make_artifacts(tmp)
        app = create_app(art, embedder=FakeEmbedder(), docs_cache_dir=tmp, frontend_dir=tmp / "no-frontend", load_async=False)
        c = TestClient(app)
        q = "where to keep emergency savings in a bank"

        h = c.get("/health").json()
        assert h["ready"] and h["error"] is None and set(h["models"]) == {"jaccard", "tfidf", "bm25", "embeddings"}, h
        assert [m["id"] for m in c.get("/api/models").json()] == ["jaccard", "tfidf", "bm25", "embeddings"]

        for model in ("jaccard", "tfidf", "bm25", "embeddings"):
            r = c.get("/api/search", params={"q": q, "model": model, "k": 3}).json()
            assert r["model"] == model and 1 <= len(r["results"]) <= 3 and r["query_known"] is True, r
            assert "query_metrics" in r and r["query_metrics"]["n_relevant"] == 2
        top = c.get("/api/search", params={"q": q, "model": "bm25", "k": 3}).json()
        assert top["results"][0]["doc_id"] == "d1" and top["results"][0]["relevant"] is True
        assert top["query_metrics"]["precision"] > 0 and top["results"][0]["snippet"]

        free = c.get("/api/search", params={"q": "stocks market", "model": "bm25"}).json()  # consulta libre
        assert free["query_known"] is False and "query_metrics" not in free and free["results"][0]["relevant"] is None

        cmp_ = c.get("/api/compare", params={"q": q, "k": 3}).json()
        assert set(cmp_["runs"]) == {"jaccard", "tfidf", "bm25", "embeddings"} and cmp_["overlap"]["bm25"]["bm25"] == len(cmp_["runs"]["bm25"]["results"])
        assert c.get("/api/doc/d1").json()["text"].startswith("Keep your emergency fund")
        assert c.get("/api/doc/nope").status_code == 404
        m = c.get("/api/metrics").json()
        assert m["order"] == ["jaccard", "tfidf", "bm25", "embeddings"] and "per_query" not in m["models"]["bm25"]
        assert isinstance(c.get("/api/sample-queries").json(), list)
        assert c.get("/api/search", params={"q": q, "model": "xyz"}).status_code == 404
        assert c.get("/api/search", params={"q": ""}).status_code == 422
        assert c.get("/api/search", params={"q": q, "k": 99}).status_code == 422
        none = c.get("/api/search", params={"q": "zzzz qqqq", "model": "bm25"}).json()  # sin términos en común
        assert none["results"] == []
    print("OK API: salud, modelos, búsqueda (4 modelos), comparación, documento, métricas y validaciones")


if __name__ == "__main__":
    main()
