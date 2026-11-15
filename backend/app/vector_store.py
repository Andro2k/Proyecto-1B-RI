"""Base de datos vectorial con FAISS. Se guarda un IndexScalarQuantizer fp16 (mitad de tamaño que
un índice plano, con pérdida de precisión despreciable) usando producto interno sobre vectores normalizados."""
import json
from pathlib import Path

import faiss
import numpy as np


def build_faiss(vectors: np.ndarray, fp16: bool = True) -> faiss.Index:
    dim = vectors.shape[1]
    if fp16:
        index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_store(index: faiss.Index, doc_ids: list[str], model_name: str, max_length: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    meta = {"model_name": model_name, "max_length": max_length, "doc_ids": doc_ids}
    (out_dir / "faiss_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_store(out_dir: Path) -> tuple[faiss.Index, dict]:
    index = faiss.read_index(str(out_dir / "faiss.index"))
    meta = json.loads((out_dir / "faiss_meta.json").read_text(encoding="utf-8"))
    return index, meta
