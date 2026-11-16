"""Textos de los documentos para mostrarlos en la web. Se guardan comprimidos en artifacts/docs.jsonl.gz
y al arrancar se vuelcan a un SQLite temporal: así los 57k textos no ocupan RAM (el plan gratuito
de Render tiene poca) y se consultan solo los que se muestran."""
import gzip
import json
import sqlite3
import tempfile
from pathlib import Path


class DocStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @classmethod
    def ensure(cls, gz_path: Path, cache_dir: Path | None = None) -> "DocStore":
        st = gz_path.stat()
        db = Path(cache_dir or tempfile.gettempdir()) / f"fiqa_docs_{st.st_size}_{int(st.st_mtime)}.sqlite"
        if not db.exists():
            tmp = db.with_suffix(".tmp")
            tmp.unlink(missing_ok=True)
            con = sqlite3.connect(tmp)
            con.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, title TEXT, text TEXT) WITHOUT ROWID")
            batch: list[tuple[str, str, str]] = []
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    batch.append((d["id"], d["title"], d["text"]))
                    if len(batch) >= 5000:
                        con.executemany("INSERT INTO docs VALUES (?, ?, ?)", batch)
                        batch = []
            if batch:
                con.executemany("INSERT INTO docs VALUES (?, ?, ?)", batch)
            con.commit()
            con.close()
            tmp.replace(db)
        return cls(db)

    def get_many(self, ids: list[str]) -> dict[str, tuple[str, str]]:
        if not ids:
            return {}
        con = sqlite3.connect(self.db_path)
        try:
            marks = ",".join("?" * len(ids))
            rows = con.execute(f"SELECT id, title, text FROM docs WHERE id IN ({marks})", list(ids)).fetchall()
        finally:
            con.close()
        return {r[0]: (r[1], r[2]) for r in rows}
