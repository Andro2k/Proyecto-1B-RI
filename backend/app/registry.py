"""Construye los modelos de recuperación a partir de los artefactos y de config.json.
Es el único lugar que decide qué parámetros usa cada modelo (lo usan los scripts y la API)."""
import json
from pathlib import Path

from .inverted_index import InvertedIndex
from .models import BM25Model, JaccardModel, TfidfModel
from .models.embeddings import EmbeddingModel

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
INDEX_PATH = ARTIFACTS / "index.pkl.gz"
CONFIG_PATH = ARTIFACTS / "config.json"

# Valores de la asignatura (notebooks 03 y 04); tune.py los reemplaza por los mejores en dev
DEFAULT_CONFIG = {
    "bm25": {"k1": 1.2, "b": 0.75},
    "tfidf": {"tf": "raw", "idf": "ln1"},
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if path.exists():
        for k, v in json.loads(path.read_text(encoding="utf-8")).items():
            cfg.setdefault(k, {}).update(v)
    return cfg


def build_models(index: InvertedIndex | None = None, artifacts: Path = ARTIFACTS) -> dict:
    """nombre -> modelo. Los embeddings se incluyen solo si ya existe la base vectorial."""
    index = index or InvertedIndex.load(artifacts / "index.pkl.gz")
    cfg = load_config(artifacts / "config.json")
    models = {
        "jaccard": JaccardModel(index),
        "tfidf": TfidfModel(index, **cfg["tfidf"]),
        "bm25": BM25Model(index, **cfg["bm25"]),
    }
    if (artifacts / "faiss.index").exists():
        models["embeddings"] = EmbeddingModel(artifacts)
    return models
