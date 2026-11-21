"""Esquemas de las respuestas de la API (se muestran en /docs y en /openapi.json).

Solo describen lo que `main.py` ya devuelve: no cambian el comportamiento de los endpoints.
"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Health(BaseModel):
    ready: bool = Field(description="True cuando los índices y modelos ya están cargados.")
    error: str | None = Field(description="Mensaje de error si la carga falló; null si todo va bien.")
    warnings: list[str] = Field(description="Avisos no fatales (p. ej. los embeddings no están disponibles).")
    models: list[str] = Field(description="Identificadores de los modelos cargados.")


class ModelInfo(BaseModel):
    id: str = Field(description="Identificador: jaccard, tfidf, bm25 o embeddings.")
    label: str = Field(description="Nombre para mostrar.")


class SearchResult(BaseModel):
    rank: int = Field(description="Posición en el ranking (1 = más relevante).")
    doc_id: str
    score: float = Field(description="Puntaje del modelo; su escala cambia entre modelos y no es comparable.")
    title: str
    snippet: str = Field(description="Inicio del texto del documento.")
    relevant: bool | None = Field(
        description="Según los qrels; null si la consulta es libre (no tiene qrels)."
    )


class QueryMetrics(BaseModel):
    k: int
    n_relevant: int = Field(description="Documentos relevantes de la consulta según los qrels.")
    precision: float = Field(description="Precision@k.")
    recall: float = Field(description="Recall@k.")
    ndcg: float = Field(description="nDCG@k.")
    ap: float = Field(description="Precisión promedio (AP) de la consulta sobre los 100 primeros resultados.")


class Run(BaseModel):
    """Ranking de un modelo para una consulta."""

    model: str
    label: str
    k: int
    latency_ms: float
    results: list[SearchResult]
    query_metrics: QueryMetrics | None = Field(
        default=None, description="Solo si la consulta tiene qrels (es una de las de prueba)."
    )


class SearchResponse(Run):
    query_known: bool = Field(description="True si la consulta es una de las de prueba (tiene qrels).")


class CompareResponse(BaseModel):
    query: str
    k: int
    query_known: bool
    runs: dict[str, Run] = Field(description="Un ranking por modelo disponible.")
    overlap: dict[str, dict[str, int]] = Field(
        description="Documentos que comparten los top-k de cada par de modelos."
    )


class DocResponse(BaseModel):
    doc_id: str
    title: str
    text: str


class ModelMetrics(BaseModel):
    label: str
    n_queries: int
    summary: dict[str, float] = Field(description="Promedios: precision@k, recall@k, ndcg@k y MAP@100.")
    latency_ms: float = Field(description="Tiempo medio por consulta durante la evaluación.")


class MetricsResponse(BaseModel):
    config: dict[str, Any] = Field(description="Parámetros usados (BM25, TF-IDF), nº de consultas y tamaño del índice.")
    models: dict[str, ModelMetrics]
    order: list[str] = Field(description="Orden sugerido para mostrar los modelos.")


class QueryCase(BaseModel):
    qid: str | None = None
    query: str
    a: float = Field(description="AP del modelo semántico (embeddings).")
    b: float = Field(description="AP de la línea base léxica (BM25).")


class Bucket(BaseModel):
    n: int
    a: float
    b: float


class AnalysisResponse(BaseModel):
    """Comparación consulta por consulta: a = embeddings, b = BM25 (medida: AP)."""

    model_config = ConfigDict(extra="allow")

    a: str
    b: str
    labels: dict[str, str]
    n: int = Field(description="Consultas comparadas.")
    mean_a: float
    mean_b: float
    a_gana: int
    b_gana: int
    empates: int
    diff_media: float = Field(description="Diferencia media de AP (a - b).")
    ic95: list[float] = Field(description="Intervalo de confianza del 95 % (bootstrap pareado) de la diferencia.")
    buckets: dict[str, Bucket] = Field(description="Resultados por tipo de consulta (corta, larga, con cifras...).")
    a_mejora_mas: list[QueryCase]
    a_empeora_mas: list[QueryCase]


class SampleQuery(BaseModel):
    query: str
    note: str
