"""Carga del corpus FiQA-2018 ya normalizado (ver scripts/download_fiqa.py)."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fiqa"


def load_corpus(data_dir: Path = DATA_DIR) -> dict[str, dict[str, str]]:
    """doc_id -> {"title": str, "text": str}"""
    corpus = {}
    with open(data_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
    return corpus


def load_queries(data_dir: Path = DATA_DIR, only_with_qrels: str | None = "test") -> dict[str, str]:
    """query_id -> texto. Por defecto solo las consultas que tienen qrels en el split indicado."""
    queries = {}
    with open(data_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["id"]] = d["text"]
    if only_with_qrels:
        qrels = load_qrels(only_with_qrels, data_dir)
        queries = {qid: t for qid, t in queries.items() if qid in qrels}
    return queries


def load_qrels(split: str = "test", data_dir: Path = DATA_DIR) -> dict[str, dict[str, int]]:
    """query_id -> {doc_id: relevancia}. Solo pares con relevancia > 0."""
    qrels: dict[str, dict[str, int]] = {}
    with open(data_dir / f"qrels_{split}.tsv", encoding="utf-8") as f:
        next(f)  # encabezado
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels
