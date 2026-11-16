"""Prueba el punto de entrada (main.py) sin ejecutar el pipeline real.
Uso: uv run python -m backend.tests.test_main"""
import contextlib
import io
import tempfile
from pathlib import Path

import main as entry


def run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = entry.main(argv)
    return code, out.getvalue()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert entry.missing_artifacts(tmp) == entry.REQUIRED
        for name in entry.REQUIRED:
            (tmp / name).write_bytes(b"x")
        assert entry.missing_artifacts(tmp) == []

    names = [n for n, _, _ in entry.pipeline_steps()]
    assert names == ["download", "stats", "index", "tune", "embeddings", "eval", "compare", "runtime"], names
    assert "embeddings" not in [n for n, _, _ in entry.pipeline_steps(["embeddings"])]

    code, out = run(["pipeline", "--dry-run", "--skip", "download", "embeddings"])
    assert code == 0 and "build_index" in out and "download_fiqa" not in out and "build_embeddings" not in out, out
    assert "--a embeddings --b bm25" in out and out.index("run_eval") < out.index("compare_models")

    args = entry.build_parser().parse_args(["serve", "--port", "9000", "--reload"])
    assert args.port == 9000 and args.reload and args.func is entry.cmd_serve
    try:
        with contextlib.redirect_stderr(io.StringIO()):  # argparse imprime el error de uso
            entry.build_parser().parse_args(["pipeline", "--skip", "inexistente"])
        raise AssertionError("debió rechazar un paso desconocido")
    except SystemExit as exc:
        assert exc.code == 2
    print("OK main.py: artefactos, pasos del pipeline, --skip/--dry-run y argumentos de serve")


if __name__ == "__main__":
    main()
