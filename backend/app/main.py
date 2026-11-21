"""API web (FastAPI) y servidor del frontend.
Local:   uv run uvicorn backend.app.main:app --reload
Render:  uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT"""
import html
import json
import random
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from .docstore import DocStore
from .evaluation import query_metrics
from .inverted_index import InvertedIndex
from .registry import ARTIFACTS, build_models
from .schemas import (AnalysisResponse, CompareResponse, DocResponse, Health, MetricsResponse, ModelId, ModelInfo,
                      SampleQuery, SearchResponse)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
LABELS = {
    "jaccard": "Binario (Jaccard)",
    "tfidf": "Vectorial (TF-IDF)",
    "bm25": "Probabilístico (BM25)",
    "embeddings": "Vectorial (Embeddings)",
}
SNIPPET_CHARS = 320
DEPTH = 100  # profundidad con la que se calculan las métricas de una consulta conocida


def _norm(text: str) -> str:
    return " ".join(text.split())


def _clean(text: str) -> str:
    """El corpus trae entidades HTML (&gt;, &amp;...): se decodifican para mostrarlas."""
    return html.unescape(text)


def _snippet(text: str) -> str:
    text = _clean(text)
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


def _release_memory() -> None:
    """Devuelve a la base del sistema la memoria que Python ya liberó (Linux); ayuda con los 512 MB de Render."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class Runtime:
    """Todo lo que se carga al arrancar. Se llena en un hilo para que /health responda de inmediato."""

    def __init__(self, artifacts: Path, embedder=None, docs_cache_dir: Path | None = None) -> None:
        self.artifacts, self.embedder, self.docs_cache_dir = artifacts, embedder, docs_cache_dir
        self.ready, self.error = False, None
        self.warnings: list[str] = []
        self.models: dict = {}
        self.docs: DocStore | None = None
        self.queries: dict[str, str] = {}
        self.qrels: dict[str, dict[str, int]] = {}
        self.qid_by_text: dict[str, str] = {}
        self.metrics: dict = {}
        self.analysis: dict = {}
        self.samples: list[dict] = []

    def load(self) -> None:
        a = self.artifacts
        try:
            index = InvertedIndex.load(a / "index.pkl.gz")
            self.models = build_models(index, a)
            if self.embedder is not None and "embeddings" in self.models:
                self.models["embeddings"]._embedder = self.embedder
            self.docs = DocStore.ensure(a / "docs.jsonl.gz", self.docs_cache_dir)
            if (a / "benchmark.json").exists():
                bench = json.loads((a / "benchmark.json").read_text(encoding="utf-8"))
                self.queries = bench["queries"]
                self.qrels = {q: {d: int(g) for d, g in r.items() if int(g) >= 1} for q, r in bench["qrels"].items()}
                self.qid_by_text = {_norm(t): q for q, t in self.queries.items()}
            if (a / "metrics.json").exists():
                self.metrics = json.loads((a / "metrics.json").read_text(encoding="utf-8"))
            if (a / "comparison_embeddings_vs_bm25.json").exists():
                self.analysis = json.loads((a / "comparison_embeddings_vs_bm25.json").read_text(encoding="utf-8"))
            self.samples = self._build_samples()
            if "embeddings" in self.models:  # carga el modelo ONNX ahora y no en la primera consulta del usuario
                try:
                    self.models["embeddings"].search("calentamiento", 1)
                except Exception as exc:  # sin modelo (red, memoria...) el resto del sistema sigue funcionando
                    self.models.pop("embeddings")
                    self.warnings.append(f"Los embeddings no están disponibles ({type(exc).__name__}: {exc}).")
            _release_memory()
            self.ready = True
        except Exception as exc:  # se muestra en /health para poder diagnosticar el despliegue
            self.error = f"{type(exc).__name__}: {exc}"

    def _build_samples(self) -> list[dict]:
        out: list[dict] = []
        cmp_path = self.artifacts / "comparison_embeddings_vs_bm25.json"
        if cmp_path.exists():
            c = json.loads(cmp_path.read_text(encoding="utf-8"))
            for r in c.get("a_mejora_mas", [])[:3]:
                out.append({"query": r["query"], "note": f"Embeddings gana (AP {r['a']:.2f} frente a {r['b']:.2f} de BM25)"})
            for r in c.get("a_empeora_mas", [])[:3]:
                out.append({"query": r["query"], "note": f"BM25 gana (AP {r['b']:.2f} frente a {r['a']:.2f} de Embeddings)"})
        rng = random.Random(7)
        qids = sorted(self.queries)
        for q in rng.sample(qids, min(4, len(qids))):
            out.append({"query": self.queries[q], "note": "Consulta de prueba"})
        return out

    # ---- utilidades de las respuestas -------------------------------------------------
    def _need_ready(self) -> None:
        if self.error:
            raise HTTPException(500, f"Error al cargar el sistema: {self.error}")
        if not self.ready:
            raise HTTPException(503, "El sistema está cargando los índices; reintenta en unos segundos.")

    def run(self, model: str, q: str, k: int) -> dict:
        if model not in self.models:
            raise HTTPException(404, f"Modelo desconocido: {model}. Disponibles: {list(self.models)}")
        q = _norm(q)
        qid = self.qid_by_text.get(q)
        t0 = time.perf_counter()
        hits = self.models[model].search(q, max(k, DEPTH) if qid else k)
        latency = (time.perf_counter() - t0) * 1000
        top = hits[:k]
        docs = self.docs.get_many([d for d, _ in top])
        rels = self.qrels.get(qid, {}) if qid else {}
        results = []
        for rank, (d, s) in enumerate(top, 1):
            title, text = docs.get(d, ("", ""))
            results.append({"rank": rank, "doc_id": d, "score": round(s, 4), "title": _clean(title),
                            "snippet": _snippet(text), "relevant": (d in rels) if qid else None})
        out = {"model": model, "label": LABELS[model], "k": k, "latency_ms": round(latency, 1), "results": results}
        if qid and rels:
            m = query_metrics([d for d, _ in hits], rels, (k,))
            out["query_metrics"] = {"k": k, "n_relevant": len(rels), "precision": m[f"precision@{k}"],
                                    "recall": m[f"recall@{k}"], "ndcg": m[f"ndcg@{k}"], "ap": m["ap"]}
        return out


TAGS = [
    {"name": "Sistema", "description": "Estado del servicio y modelos disponibles."},
    {"name": "Búsqueda", "description": "Consultas de texto libre y comparación entre modelos."},
    {"name": "Documentos", "description": "Acceso al texto de los documentos del corpus."},
    {"name": "Evaluación", "description": "Métricas de recuperación y análisis comparativo."},
]
ERR_LOADING = {503: {"description": "El sistema todavía está cargando los índices; reintentar en unos segundos."}}


def create_app(artifacts: Path = ARTIFACTS, embedder=None, docs_cache_dir: Path | None = None,
               frontend_dir: Path = FRONTEND_DIR, load_async: bool = True) -> FastAPI:
    app = FastAPI(
        title="Recuperación de Información · FiQA-2018",
        description=("Compara cuatro modelos de recuperación (Jaccard binario, TF-IDF con coseno, BM25 y embeddings con FAISS) "
                     "sobre el corpus FiQA-2018. Todos los endpoints son de solo lectura (GET)."),
        openapi_tags=TAGS,
    )
    rt = Runtime(artifacts, embedder, docs_cache_dir)
    app.state.rt = rt
    if load_async:
        threading.Thread(target=rt.load, daemon=True).start()
    else:
        rt.load()

    @app.get("/health", response_model=Health, tags=["Sistema"], summary="Estado del servicio")
    def health():
        """Indica si los índices ya cargaron y qué modelos están disponibles. Responde aunque la carga siga en curso."""
        return {"ready": rt.ready, "error": rt.error, "warnings": rt.warnings, "models": list(rt.models)}

    @app.get("/api/models", response_model=list[ModelInfo], tags=["Sistema"], summary="Modelos disponibles",
             responses=ERR_LOADING)
    def models():
        """Lista los modelos que se pueden consultar (los embeddings faltan si su modelo no pudo cargarse)."""
        rt._need_ready()
        return [{"id": m, "label": LABELS[m]} for m in LABELS if m in rt.models]

    @app.get("/api/search", response_model=SearchResponse, response_model_exclude_unset=True, tags=["Búsqueda"],
             summary="Consultar un modelo", responses={**ERR_LOADING, 404: {"description": "Modelo válido pero no cargado (p. ej. embeddings sin su modelo)."}})
    def search(q: str = Query(..., min_length=1, max_length=500, description="Consulta en texto libre."),
               model: ModelId = Query("bm25", description="Modelo a consultar."),
               k: int = Query(10, ge=1, le=50, description="Cantidad de resultados (top-k).")):
        """Devuelve el ranking del modelo elegido. Si la consulta es una de las de prueba, incluye `relevant` por
        documento y `query_metrics` (Precision, Recall, nDCG y AP) calculadas con los qrels."""
        rt._need_ready()
        out = rt.run(model, q, k)
        out["query_known"] = _norm(q) in rt.qid_by_text
        return out

    @app.get("/api/compare", response_model=CompareResponse, response_model_exclude_unset=True, tags=["Búsqueda"],
             summary="Comparar todos los modelos", responses=ERR_LOADING)
    def compare(q: str = Query(..., min_length=1, max_length=500, description="Consulta en texto libre."),
                k: int = Query(10, ge=1, le=50, description="Cantidad de resultados (top-k).")):
        """Ejecuta la misma consulta en todos los modelos disponibles y calcula cuántos documentos comparten sus top-k."""
        rt._need_ready()
        runs = {m: rt.run(m, q, k) for m in LABELS if m in rt.models}
        ids = {m: {r["doc_id"] for r in run["results"]} for m, run in runs.items()}
        overlap = {a: {b: len(ids[a] & ids[b]) for b in ids} for a in ids}
        return {"query": _norm(q), "k": k, "query_known": _norm(q) in rt.qid_by_text, "runs": runs, "overlap": overlap}

    @app.get("/api/doc/{doc_id}", response_model=DocResponse, tags=["Documentos"], summary="Texto completo de un documento",
             responses={**ERR_LOADING, 404: {"description": "Documento no encontrado."}})
    def doc(doc_id: str):
        """Devuelve el título y el texto completo de un documento del corpus."""
        rt._need_ready()
        found = rt.docs.get_many([doc_id])
        if doc_id not in found:
            raise HTTPException(404, "Documento no encontrado")
        title, text = found[doc_id]
        return {"doc_id": doc_id, "title": _clean(title), "text": _clean(text)}

    @app.get("/api/metrics", response_model=MetricsResponse, tags=["Evaluación"], summary="Métricas globales por modelo",
             responses={**ERR_LOADING, 404: {"description": "Aún no se ejecutó la evaluación."}})
    def metrics():
        """Promedios sobre las consultas de prueba (Precision, Recall, nDCG y MAP@100) y latencia media por modelo."""
        rt._need_ready()
        if not rt.metrics:
            raise HTTPException(404, "Aún no hay métricas: ejecuta backend.scripts.run_eval")
        models = {m: {"label": LABELS[m], "n_queries": v["n_queries"], "summary": v["summary"], "latency_ms": v["latency_ms"]}
                  for m, v in rt.metrics.items() if not m.startswith("_") and m in LABELS}
        return {"config": rt.metrics.get("_config", {}), "models": models, "order": [m for m in LABELS if m in models]}

    @app.get("/api/analysis", response_model=AnalysisResponse, tags=["Evaluación"], summary="Embeddings frente a BM25",
             responses={**ERR_LOADING, 404: {"description": "Aún no se generó el análisis."}})
    def analysis():
        """Comparación consulta por consulta (AP): victorias de cada modelo, diferencia media con IC 95 %, resultados
        por tipo de consulta y los casos donde la recuperación semántica más mejora o empeora."""
        rt._need_ready()
        if not rt.analysis:
            raise HTTPException(404, "Aún no hay análisis: ejecuta backend.scripts.compare_models")
        # a = modelo semántico, b = línea base léxica; "a_mejora_mas"/"a_empeora_mas" se miden con AP
        return {"a": "embeddings", "b": "bm25", "labels": {"a": LABELS["embeddings"], "b": LABELS["bm25"]}, **rt.analysis}

    @app.get("/api/sample-queries", response_model=list[SampleQuery], tags=["Búsqueda"],
             summary="Consultas de ejemplo", responses=ERR_LOADING)
    def sample_queries():
        """Consultas de prueba con qrels, elegidas para mostrar dónde gana cada modelo."""
        rt._need_ready()
        return rt.samples

    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


_default_app: FastAPI | None = None


def __getattr__(name: str):
    """`uvicorn backend.app.main:app` pide `app`: se crea entonces y no al importar el módulo
    (así los tests y scripts que solo importan create_app no cargan los índices en segundo plano)."""
    global _default_app
    if name == "app":
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(name)
