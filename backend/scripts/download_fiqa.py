"""Descarga FiQA-2018 (BeIR) desde Hugging Face y lo normaliza a:
    backend/data/fiqa/corpus.jsonl      {"id","title","text"}
    backend/data/fiqa/queries.jsonl     {"id","text"}
    backend/data/fiqa/qrels_<split>.tsv query-id \t corpus-id \t score

Uso (desde la raíz del proyecto):  uv run python -m backend.scripts.download_fiqa
"""
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "fiqa"
RAW_DIR = OUT_DIR.parent / "raw"
DATA_EXT = (".parquet", ".jsonl", ".jsonl.gz", ".json", ".tsv", ".csv")


def _read(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith((".jsonl", ".jsonl.gz")):
        return pd.read_json(path, lines=True)
    if name.endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path, sep="\t" if name.endswith(".tsv") else ",")


def _files(root: Path, keyword: str) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name.lower().endswith(DATA_EXT) and keyword in str(p.relative_to(root)).lower()
    )


def _concat(root: Path, keyword: str) -> pd.DataFrame:
    files = _files(root, keyword)
    if not files:
        raise FileNotFoundError(f"No encontré archivos '{keyword}' en {root}")
    return pd.concat([_read(p) for p in files], ignore_index=True)


def _id_col(df: pd.DataFrame) -> str:
    return "_id" if "_id" in df.columns else "id"


def normalize(raw_corpus: Path, raw_qrels: Path, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = _concat(raw_corpus, "corpus")
    c = pd.DataFrame({
        "id": corpus[_id_col(corpus)].astype(str),
        "title": corpus.get("title", "").fillna(""),
        "text": corpus["text"].fillna(""),
    })
    c.to_json(out_dir / "corpus.jsonl", orient="records", lines=True, force_ascii=False)

    queries = _concat(raw_corpus, "queries")
    q = pd.DataFrame({"id": queries[_id_col(queries)].astype(str), "text": queries["text"]})
    q.to_json(out_dir / "queries.jsonl", orient="records", lines=True, force_ascii=False)

    for split, aliases in {"train": ("train",), "dev": ("dev", "validation"), "test": ("test",)}.items():
        files = [p for a in aliases for p in _files(raw_qrels, a)]
        if not files:
            continue
        df = pd.concat([_read(p) for p in files], ignore_index=True)
        df = df.rename(columns={"query_id": "query-id", "corpus_id": "corpus-id"})
        df = df[["query-id", "corpus-id", "score"]].astype({"query-id": str, "corpus-id": str})
        df.to_csv(out_dir / f"qrels_{split}.tsv", sep="\t", index=False)
        print(f"qrels_{split}.tsv: {len(df)} pares")
    print(f"corpus: {len(c)} docs | queries: {len(q)}")


def main() -> None:
    raw_corpus = Path(snapshot_download("BeIR/fiqa", repo_type="dataset", local_dir=RAW_DIR / "fiqa"))
    raw_qrels = Path(snapshot_download("BeIR/fiqa-qrels", repo_type="dataset", local_dir=RAW_DIR / "fiqa-qrels"))
    normalize(raw_corpus, raw_qrels)


if __name__ == "__main__":
    main()
