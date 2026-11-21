# Sistema de Recuperación de Información · FiQA-2018

**Proyecto de 1er Bimestre · Recuperación de Información** — Prof. Iván Carrera · Escuela Politécnica Nacional
**Fecha de entrega:** 23 de noviembre de 2026

**Integrantes:** Danny Molina · J. Luis Andino · Luis Morocho

**URL del despliegue:** _pendiente — se completa al publicar la web (ver [Despliegue](#despliegue))_

---

## ¿Qué hace?

Sistema de recuperación de información sobre el corpus **FiQA-2018** (57 638 respuestas de foros financieros, 648 consultas de prueba con sus qrels) que permite **comparar cuatro modelos** desde una web:

| Modelo | Implementación |
|---|---|
| Binario | Similitud de **Jaccard** sobre vectores binarios, calculada con el índice invertido |
| Vectorial (TF-IDF) | **Coseno** entre vectores TF-IDF (implementación propia sobre el índice invertido) |
| Probabilístico | **BM25** (implementación propia) |
| Vectorial (Embeddings) | `all-MiniLM-L6-v2` (fastembed/ONNX) + búsqueda vectorial en **FAISS** |

La web empieza con una **portada minimalista**: una barra de búsqueda y, en la esquina, el acceso a **Análisis**. Los **ajustes avanzados** (modelo, número de resultados y consultas de ejemplo) se despliegan bajo la barra. Al buscar se muestra el ranking de un modelo o, por defecto, la **comparación de los cuatro lado a lado** (en columnas o como tabla de posiciones). Si la consulta es una de las de prueba, se marcan los documentos relevantes según los qrels y se calculan Precision, Recall, nDCG y AP de esa consulta. La página **Análisis** reúne las métricas de evaluación y la comparación entre embeddings y BM25 (victorias, intervalo de confianza, resultados por tipo de consulta y casos extremos).

## Resultados

Evaluación sobre las **648 consultas de prueba** (los parámetros se eligieron en las 500 consultas de desarrollo; ver `backend/artifacts/tuning.json`). Reproducible con `uv run python -m backend.scripts.run_eval`.

| Modelo | Precision@10 | Recall@10 | nDCG@10 | Recall@20 | MAP@100 | ms/consulta* |
|---|---:|---:|---:|---:|---:|---:|
| Binario (Jaccard) | 0.0154 | 0.0623 | 0.0490 | 0.0869 | 0.0373 | 1.3 |
| Vectorial (TF-IDF) | 0.0525 | 0.2193 | 0.1729 | 0.2963 | 0.1405 | 1.2 |
| Probabilístico (BM25) | 0.0671 | 0.3052 | 0.2474 | 0.3862 | 0.2042 | 0.6 |
| **Vectorial (Embeddings)** | **0.1045** | **0.4413** | **0.3687** | **0.5194** | **0.3096** | 58.4 |

\* Tiempo medio por consulta en la máquina de desarrollo; el de embeddings incluye calcular el embedding de la consulta.

**Embeddings frente a BM25** (`uv run python -m backend.scripts.compare_models --a embeddings --b bm25`): diferencia media de AP de +0.1054 (IC 95 % bootstrap pareado: +0.083 a +0.128). Embeddings gana en 393 consultas, BM25 en 147 y empatan 108. La ventaja es mayor en consultas largas (+0.124) que en cortas (+0.073). BM25 gana sobre todo con consultas muy breves o de términos concretos (p. ej. «Bitcoin Cost Basis Purchases», «How does pre-market trading work?»); embeddings gana cuando la consulta y la respuesta no comparten vocabulario.

Nota: Precision@10 es baja para todos porque cada consulta tiene pocos documentos relevantes (2.6 en promedio) y los qrels de FiQA son incompletos.

## Cumplimiento del enunciado

| Requisito | Dónde está |
|---|---|
| Índice invertido con frecuencias; tokenización, normalización y stopwords | `backend/app/preprocessing.py`, `backend/app/inverted_index.py` |
| Jaccard (vectores binarios), TF-IDF con coseno y BM25 | `backend/app/models/jaccard.py`, `tfidf.py`, `bm25.py` |
| Embeddings con modelo preentrenado y base vectorial | `backend/app/embedder.py`, `vector_store.py` (FAISS), `models/embeddings.py` |
| Interfaz web: consultas, elegir modelo, top-n, comparar, métricas | `frontend/` + API en `backend/app/main.py` |
| Evaluación con qrels: Precision y Recall por consulta, MAP del sistema | `backend/app/evaluation.py`, `backend/scripts/run_eval.py` |
| Comparación de modelos y análisis por tipo de consulta | `backend/scripts/compare_models.py` (salida en `backend/artifacts/comparison_embeddings_vs_bm25.json`) |
| Sin motores de búsqueda completos (Elasticsearch, Solr, Whoosh) | Los modelos clásicos son propios y se verifican con los ejemplos de las diapositivas de la asignatura (`backend/tests/test_class_examples.py`) |
| Informe técnico (máx. 5 páginas) | _pendiente_ |
| URL del despliegue | _pendiente_ |

## Estructura del repositorio

```
main.py                  punto de entrada: serve | check | test | pipeline
backend/
  app/                   preprocesamiento, índice, modelos, evaluación, API (FastAPI)
  scripts/               descarga, índice, tuning, embeddings, evaluación, comparación, datos de la web
  tests/                 pruebas (se ejecutan con `main.py test`)
  artifacts/             índices y resultados ya generados (se versionan para poder desplegar)
frontend/                interfaz web (HTML, CSS y JS sin build)
render.yaml, Dockerfile  despliegue
```

## Requisitos

- **Python 3.12** (fijado en `.python-version`) y [uv](https://docs.astral.sh/uv/).
- Conexión a Internet la primera vez: se descarga el modelo ONNX de embeddings (~90 MB) a `backend/models_cache/`.

```powershell
# Windows (PowerShell)
uv python install 3.12
uv sync
```
```bash
# Linux / macOS
uv python install 3.12 && uv sync
```

## Inicio rápido (con los artefactos del repositorio)

El repositorio incluye los índices y resultados ya generados, así que no hace falta recalcular nada:

```powershell
uv run python main.py check          # verifica que estén los artefactos
uv run python main.py serve --open   # levanta la web en http://127.0.0.1:8000 y abre el navegador
```

La primera carga tarda unos segundos (índices y modelo en memoria). Mientras tanto la página muestra «Cargando índices y modelos…». Opciones útiles: `--port 9000`, `--reload` (desarrollo), `--host 0.0.0.0`.

> En Windows puede aparecer un aviso de `huggingface_hub` sobre enlaces simbólicos; es inofensivo (se silencia con `$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1`).

## Reproducir todo el proyecto desde cero

Todo el pipeline en un solo comando (el paso de embeddings tarda varios minutos en CPU):

```powershell
uv run python main.py pipeline                       # ejecuta los 8 pasos en orden
uv run python main.py pipeline --dry-run             # solo muestra los pasos
uv run python main.py pipeline --skip download embeddings   # omite pasos ya hechos
```

O paso a paso:

| # | Comando | Qué hace | Genera |
|---|---|---|---|
| 1 | `uv run python -m backend.scripts.download_fiqa` | Descarga FiQA-2018 (BeIR) y lo normaliza | `backend/data/fiqa/` (no se versiona) |
| 2 | `uv run python -m backend.scripts.corpus_stats` | Estadísticas del corpus (para el informe) | salida en consola |
| 3 | `uv run python -m backend.scripts.build_index` | Preprocesa el corpus y construye el índice invertido | `artifacts/index.pkl.gz` |
| 4 | `uv run python -m backend.scripts.tune` | Elige parámetros de BM25 y TF-IDF en las consultas de **desarrollo** | `artifacts/config.json`, `tuning.json` |
| 5 | `uv run python -m backend.scripts.build_embeddings` | Embeddings del corpus y base FAISS (reanudable) | `artifacts/faiss.index`, `faiss_meta.json` |
| 6 | `uv run python -m backend.scripts.run_eval` | Evalúa los modelos en las consultas de **prueba** | `artifacts/metrics.json` |
| 7 | `uv run python -m backend.scripts.compare_models --a embeddings --b bm25` | Comparación por consulta y por tipo de consulta | `artifacts/comparison_embeddings_vs_bm25.json` |
| 8 | `uv run python -m backend.scripts.build_runtime_data` | Textos y qrels que necesita la web | `artifacts/docs.jsonl.gz`, `benchmark.json` |

Para probar rápido los embeddings: `uv run python -m backend.scripts.build_embeddings --limit 2000`. Para ver el top-5 de cada modelo en consola: `uv run python -m backend.scripts.try_models "tu consulta"`.

Después de regenerar los artefactos, súbelos al repositorio (`git add backend/artifacts`) para que el despliegue use los mismos.

## Pruebas

```powershell
uv run python main.py test
```

Incluye la verificación de los modelos contra implementaciones ingenuas, la **reproducción de los ejemplos numéricos de las diapositivas** (BM25, TF-IDF y NDCG), las métricas, la capa FAISS, la API y el punto de entrada.

## API

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado de carga y avisos (p. ej. embeddings no disponibles) |
| `GET /api/models` | Modelos disponibles |
| `GET /api/search?q=...&model=bm25&k=10` | Top-k de un modelo (`jaccard`, `tfidf`, `bm25`, `embeddings`; cualquier otro valor responde 422) |
| `GET /api/compare?q=...&k=10` | Los cuatro modelos y el solape entre sus top-k |
| `GET /api/doc/{id}` | Texto del documento |
| `GET /api/metrics` | Métricas globales, latencia y configuración |
| `GET /api/analysis` | Comparación embeddings frente a BM25 (resultado de `compare_models`) |
| `GET /api/sample-queries` | Consultas de ejemplo |

La documentación interactiva de FastAPI queda en `/docs`.

## Despliegue

La web y la API se sirven desde un solo servicio (FastAPI también entrega `frontend/`), así que se necesita una única URL.

**Render (plan gratuito)** — crea un *Blueprint* apuntando a este repositorio; `render.yaml` ya define:
- build: `pip install -r requirements.txt && python -m backend.scripts.prefetch_model` (descarga el modelo ONNX);
- start: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`;
- health check: `/health`.

El plan gratuito de Render tiene **512 MB de RAM**. Sin el modelo ONNX la aplicación ocupa unos 290 MB; si el servicio se queda sin memoria, la API sigue funcionando con los tres modelos léxicos y avisa de que los embeddings no están disponibles.

**Docker / Hugging Face Spaces** — el `Dockerfile` construye la imagen (descarga el modelo en el build) y escucha en `$PORT` (7860 por defecto). En un Space de tipo Docker, añade al `README.md` del Space el encabezado `sdk: docker` y `app_port: 7860`. Los Spaces gratuitos ofrecen bastante más RAM que Render.

Tras desplegar, escribe la URL al inicio de este README.

## Decisiones de diseño

- **Preprocesamiento único** para documentos y consultas: minúsculas, tokens alfanuméricos, 198 stopwords en inglés y stemming Snowball (`py-rust-stemmers`, elegido en lugar de NLTK porque este último consume ~150 MB de RAM solo al importarse). Los embeddings usan el texto original, sin stemming ni stopwords.
- **Modelos clásicos propios** sobre un índice invertido (`término → documentos y frecuencias`), guardado en arreglos contiguos para reducir la memoria. Las variantes de TF-IDF y los parámetros de BM25 siguen las convenciones de la asignatura y se verifican con sus ejemplos numéricos.
- **Parámetros elegidos en desarrollo, no en prueba:** `tune.py` explora k1 y b de BM25 y las variantes de TF-IDF en las consultas de desarrollo. La mejora de BM25 sobre los valores de clase (k1 = 1.2, b = 0.75) es pequeña (~0.003 de MAP@100 en desarrollo).
- **Embeddings:** `all-MiniLM-L6-v2` (384 dimensiones), vectores normalizados y FAISS con producto interno (equivale al coseno). El índice se guarda en fp16 (~44 MB) y los textos se truncan a 256 tokens.
- **Evaluación:** relevante si `rel ≥ 1`; Precision@k, Recall@k y nDCG@k por consulta, y MAP@100 para el sistema. Los documentos sin juicio cuentan como no relevantes; las convenciones están documentadas en `backend/app/evaluation.py`.
- **La web compara posiciones y documentos, no scores**: BM25 y el coseno usan escalas distintas y no son comparables entre sí.

## Limitaciones

- Los qrels de FiQA son incompletos: un documento sin juicio no es necesariamente irrelevante, lo que penaliza a los modelos que recuperan documentos útiles no juzgados.
- La búsqueda vectorial es exhaustiva (todos los documentos en cada consulta), adecuada para 57 mil documentos pero no para colecciones mucho mayores.
- El corpus y las consultas están en inglés; los modelos léxicos no resuelven sinónimos ni erratas.

---

<details>
<summary><b>Enunciado original del proyecto</b></summary>

### Objetivo
Diseñar e implementar un sistema de recuperación de información que permita comparar modelos clásicos
y modernos de recuperación.  El sistema deberá indexar un corpus textual utilizando un índice invertido
para modelos basados en términos, y una base de datos vectorial para recuperación semántica basada en
embeddings. El sistema deberá permitir ejecutar consultas de texto libre, mostrar rankings de documentos
y evaluar la calidad de los resultados mediante métricas de Recuperación de Información.

### Alcance y funcionalidades mínimas
El sistema debe cumplir con los siguientes requerimientos:
#### a. Construcción del índice
- Leer un corpus de documentos en texto plano.
- Procesamiento básico: tokenización, normalización y remoción de stopwords.
- Construcción de un índice invertido que almacene, para cada término, los documentos en los que
aparece y su frecuencia.
#### b. Modelo de recuperación
- Implementar recuperación basada en similitud Jaccard utilizando vectores binarios
- Implementar recuperación basada en similitud de coseno utilizando TF-IDF
- Implementar recuperación con BM25.
- Permitir la ejecución de consultas de texto libre.
- Mostrar un ranking de documentos ordenados por relevancia.
#### c. Recuperación semántica con embeddings
- Generar embeddings para los documentos del corpus utilizando un modelo preentrenado.
- Generar embeddings para las consultas de texto libre.
- Almacenar los embeddings en una base de datos vectorial, como ChromaDB o FAISS.
- Recuperar los documentos más similares usando búsqueda vectorial.
- Mostrar un ranking de resultados basado en similitud vectorial.
#### d. Interfaz básica
- Interfaz web de recuperación de consultas, desplegada en un servicio gratuito (Render, Vercel,
etc.)
- Funcionalidades mínimas:
> - Realizar consultas
> - Elegir el modelo de recuperación (Binario, Vectorial con TF-IDF, Probabilístico, Vectorial con Embeddings)
> - Visualizar el top-n de resultados en los diferentes modelos.
> - Comparar resultados entre los diferentes modelos.
> - Visualizar las métricas de evaluación de la recuperación.
#### e. Evaluación de resultados
- Usar un conjunto de consultas de prueba y documentos relevantes (qrels).
- Calcular para cada consulta:
> - Precision
> - Recall
- Calcular para todo el sistema:
> - MAP
#### f. Comparación de modelos
- Comparar los resultados obtenidos con los diferentes modelos de recuperación: Binario, Vectorial con TF-IDF, Probabilístico, Vectorial con Embeddings.
- Analizar en qué tipos de consultas funciona mejor cada modelo.
- Identificar casos donde la recuperación semántica mejora o empeora respecto a BM25 o TF-IDF.
### Requisitos técnicos
- Lenguaje sugerido: Python.
- Se pueden usar bibliotecas como NumPy, pandas, NLTK, spaCy, scikit-learn, sentence-transformers.
- Para recuperación vectorial se debe usar ChromaDB o FAISS.
- No se permite el uso de motores completos de búsqueda como Elasticsearch, Solr o Whoosh.
- La implementación de los modelos clásicos debe ser propia o claramente explicada.
### Entregables
- URL del despliegue del sistema.
- Código fuente con instrucciones claras de ejecución (README).
- Informe técnico (máximo 5 páginas), que incluya:
- Descripción del corpus utilizado.
- Explicación de las decisiones de diseño.
- Ejemplos de consultas y resultados.
- Análisis de métricas de evaluación.

</details>
