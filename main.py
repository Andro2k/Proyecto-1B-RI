"""Punto de entrada del proyecto: sistema de Recuperación de Información sobre FiQA-2018.

    uv run python main.py                 # levanta la web en http://127.0.0.1:8000
    uv run python main.py serve --open --reload
    uv run python main.py check           # verifica que existan los artefactos
    uv run python main.py test            # corre todas las pruebas
    uv run python main.py pipeline        # reconstruye todo: descarga, índices, embeddings y evaluación
    uv run python main.py pipeline --skip download embeddings --dry-run
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "backend" / "artifacts"

# Lo mínimo que necesita la web para arrancar (se generan con el pipeline)
REQUIRED = ["index.pkl.gz", "faiss.index", "faiss_meta.json", "docs.jsonl.gz", "benchmark.json", "metrics.json"]

# (nombre, módulo, argumentos, descripción) en el orden en que deben ejecutarse
PIPELINE = [
    ("download", "backend.scripts.download_fiqa", [], "Descargar FiQA-2018 (corpus, consultas y qrels)"),
    ("stats", "backend.scripts.corpus_stats", [], "Estadísticas del corpus (para el informe)"),
    ("index", "backend.scripts.build_index", [], "Construir el índice invertido"),
    ("tune", "backend.scripts.tune", [], "Elegir parámetros de BM25 y TF-IDF con las consultas de desarrollo"),
    ("embeddings", "backend.scripts.build_embeddings", [], "Calcular embeddings y construir la base FAISS (tarda varios minutos)"),
    ("eval", "backend.scripts.run_eval", [], "Evaluar los modelos con las consultas de prueba"),
    ("compare", "backend.scripts.compare_models", ["--a", "embeddings", "--b", "bm25"], "Comparar embeddings contra BM25 por consulta"),
    ("runtime", "backend.scripts.build_runtime_data", [], "Generar los textos y qrels que usa la web"),
]
TESTS = ["test_models", "test_class_examples", "test_evaluation", "test_vector_store", "test_api", "test_main"]


def missing_artifacts(artifacts: Path = ARTIFACTS) -> list[str]:
    return [name for name in REQUIRED if not (artifacts / name).exists()]


def pipeline_steps(skip: list[str] | None = None) -> list[tuple[str, list[str], str]]:
    skip = set(skip or [])
    return [(name, [sys.executable, "-m", module, *args], desc) for name, module, args, desc in PIPELINE if name not in skip]


def run_commands(steps: list[tuple[str, list[str], str]], dry_run: bool = False) -> int:
    for i, (name, cmd, desc) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name}: {desc}\n    $ {' '.join(cmd[1:])}", flush=True)
        if dry_run:
            continue
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            print(f"\nFalló el paso '{name}'. Corrígelo y vuelve a ejecutar; puedes saltar los pasos ya hechos con --skip.")
            return 1
    return 0


def _open_browser_when_ready(url: str) -> None:
    for _ in range(120):
        try:
            if urllib.request.urlopen(url + "/health", timeout=2).status == 200:
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(1)


def cmd_serve(args) -> int:
    missing = missing_artifacts()
    if missing and not args.force:
        print("Faltan artefactos en backend/artifacts: " + ", ".join(missing))
        print("Ejecuta `uv run python main.py pipeline` (o consulta el README) para generarlos.")
        return 1
    import uvicorn

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"Sistema de Recuperación de Información en {url}  (Ctrl+C para detener)")
    if args.open:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    uvicorn.run("backend.app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_check(_args) -> int:
    ok = True
    for name in REQUIRED:
        path = ARTIFACTS / name
        exists = path.exists()
        ok &= exists
        size = f"{path.stat().st_size / 1e6:.1f} MB" if exists else "FALTA"
        print(f"  {'OK ' if exists else 'ERR'} {name:<22} {size}")
    print("Todo listo para servir la web." if ok else "Faltan artefactos: ejecuta `main.py pipeline`.")
    return 0 if ok else 1


def cmd_test(_args) -> int:
    return run_commands([(t, [sys.executable, "-m", f"backend.tests.{t}"], "") for t in TESTS])


def cmd_pipeline(args) -> int:
    return run_commands(pipeline_steps(args.skip), dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    deployed = "PORT" in os.environ  # en Render/Spaces hay que escuchar en todas las interfaces
    parser = argparse.ArgumentParser(description="Sistema de Recuperación de Información (FiQA-2018)")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="levantar la web y la API (por defecto)")
    serve.add_argument("--host", default="0.0.0.0" if deployed else "127.0.0.1")
    serve.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    serve.add_argument("--reload", action="store_true", help="recargar al cambiar el código (desarrollo)")
    serve.add_argument("--open", action="store_true", help="abrir el navegador cuando el sistema esté listo")
    serve.add_argument("--force", action="store_true", help="arrancar aunque falten artefactos")
    serve.set_defaults(func=cmd_serve)

    sub.add_parser("check", help="verificar los artefactos necesarios").set_defaults(func=cmd_check)
    sub.add_parser("test", help="ejecutar todas las pruebas").set_defaults(func=cmd_test)

    pipe = sub.add_parser("pipeline", help="reconstruir datos, índices, embeddings y evaluación")
    pipe.add_argument("--skip", nargs="+", default=[], choices=[p[0] for p in PIPELINE], metavar="PASO",
                      help="pasos a omitir: " + ", ".join(p[0] for p in PIPELINE))
    pipe.add_argument("--dry-run", action="store_true", help="mostrar los pasos sin ejecutarlos")
    pipe.set_defaults(func=cmd_pipeline)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["serve", *argv]  # `python main.py` y `python main.py --reload` levantan la web
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
