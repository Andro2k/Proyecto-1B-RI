"""Calcula embeddings del corpus y construye la base vectorial FAISS en backend/artifacts/
Es reanudable: los lotes ya calculados se guardan en backend/data/emb_chunks/.
Uso: uv run python -m backend.scripts.build_embeddings [--model NOMBRE] [--max-length N] [--limit N]"""
import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from backend.app.data import load_corpus
from backend.app.embedder import DEFAULT_MAX_LENGTH, DEFAULT_MODEL, FastEmbedder
from backend.app.vector_store import build_faiss, save_store

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
CHUNK_DIR = Path(__file__).resolve().parents[1] / "data" / "emb_chunks"
CHUNK = 2000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--limit", type=int, default=None, help="solo los primeros N docs (para probar rápido)")
    args = ap.parse_args()

    corpus = load_corpus()
    items = list(corpus.items())[: args.limit]
    # Se ordenan por longitud para que cada lote tenga textos parecidos (menos relleno = más rápido)
    items.sort(key=lambda kv: len(kv[1]["title"]) + len(kv[1]["text"]))
    doc_ids = [d for d, _ in items]
    texts = [f"{d['title']} {d['text']}".strip() for _, d in items]

    embedder = FastEmbedder(args.model, args.max_length)
    tag = args.model.replace("/", "_") + f"_{args.max_length}_{len(items)}"
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    parts, t0 = [], time.time()
    for start in tqdm(range(0, len(texts), CHUNK), desc="Embeddings", unit="lote"):
        f = CHUNK_DIR / f"{tag}_{start:06d}.npy"
        if f.exists():
            parts.append(np.load(f))
            continue
        vecs = embedder.embed_documents(texts[start : start + CHUNK])
        np.save(f, vecs)
        parts.append(vecs)
    vectors = np.vstack(parts).astype(np.float32)
    print(f"Vectores: {vectors.shape} en {time.time() - t0:.0f}s")

    index = build_faiss(vectors)
    save_store(index, doc_ids, args.model, args.max_length, ARTIFACTS)
    size = (ARTIFACTS / "faiss.index").stat().st_size / 1e6
    print(f"FAISS guardado en {ARTIFACTS} (faiss.index: {size:.1f} MB)")


if __name__ == "__main__":
    main()
