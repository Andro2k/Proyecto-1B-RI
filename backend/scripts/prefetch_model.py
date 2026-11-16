"""Descarga el modelo de embeddings a backend/models_cache (se ejecuta en el build del despliegue).
Uso: python -m backend.scripts.prefetch_model"""
import json

from backend.app.embedder import CACHE_DIR, FastEmbedder
from backend.app.vector_store import load_store
from backend.app.registry import ARTIFACTS


def main() -> None:
    meta = load_store(ARTIFACTS)[1]
    emb = FastEmbedder(meta["model_name"], meta["max_length"])
    dim = emb.embed_query("prueba").shape[1]
    print(json.dumps({"model": meta["model_name"], "dim": dim, "cache": str(CACHE_DIR)}))


if __name__ == "__main__":
    main()
