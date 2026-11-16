# Proyecto de 1er Bimestre - Sistema de Recuperación de Información
### Prof.  Iván Carrera — Recuperación de Información
### Fecha de entrega: 23 de noviembre de 2026
### Integrantes
- Danny Molina
- J. Luis Andino

## Objetivo
Diseñar e implementar un sistema de recuperación de información que permita comparar modelos clásicos
y  modernos  de  recuperación.   El  sistema  deberá  indexar  un  corpus  textual  utilizando  un índice  invertido
para modelos basados en términos,  y una base de datos vectorial para recuperación semántica basada en
embeddings.  El sistema deberá permitir ejecutar consultas de texto libre, mostrar rankings de documentos
y evaluar la calidad de los resultados mediante métricas de Recuperación de Información.

## Alcance y funcionalidades mínimas
El sistema debe cumplir con los siguientes requerimientos:
### a. Construcción del índice
-  Leer un corpus de documentos en texto plano.
-  Procesamiento básico:  tokenización, normalización y remoción de stopwords.
-  Construcción de un índice invertido que almacene, para cada término, los documentos en los que
aparece y su frecuencia.
### b.  Modelo de recuperación
-  Implementar recuperación basada en similitud Jaccard utilizando vectores binarios
-  Implementar recuperación basada en similitud de coseno utilizando TF-IDF
-  Implementar recuperación con BM25.
-  Permitir la ejecución de consultas de texto libre.
-  Mostrar un ranking de documentos ordenados por relevancia.
### c.  Recuperación semántica con embeddings
-  Generar embeddings para los documentos del corpus utilizando un modelo preentrenado.
-  Generar embeddings para las consultas de texto libre.
-  Almacenar los embeddings en una base de datos vectorial, como ChromaDB o FAISS.
-  Recuperar los documentos más similares usando b ́usqueda vectorial.
-  Mostrar un ranking de resultados basado en similitud vectorial.
### d.  Interfaz básica
-  Interfaz  web  de  recuperación  de  consultas,  desplegada  en  un  servicio  gratuito  (Render,  Vercel,
etc.)
-  Funcionalidades mínimas:
> - Realizar consultas
> - Elegir  el  modelo  de  recuperación  (Binario,  Vectorial  con  TF-IDF,  Probabilístico,  Vectorial con Embeddings)
> - Visualizar el top-n de resultados en los diferentes modelos.
> - Comparar resultados entre los diferentes modelos.
> - Visualizar las métricas de evaluación de la recuperación.
### e.  Evaluación de resultados
-  Usar un conjunto de consultas de prueba y documentos relevantes (qrels).
-  Calcular para cada consulta:
> - Precision
> - Recall
-  Calcular para todo el sistema:
> - MAP
### f.  Comparación de modelos
-  Comparar los resultados obtenidos con los diferentes modelos de recuperación:  Binario, Vectorial con TF-IDF, Probabilístico, Vectorial con Embeddings.
-  Analizar en qué tipos de consultas funciona mejor cada modelo.
-  Identificar casos donde la recuperación semántica mejora o empeora respecto a BM25 o TF-IDF.
## Requisitos técnicos
-  Lenguaje sugerido:  Python.
-  Se pueden usar bibliotecas como NumPy, pandas, NLTK, spaCy, scikit-learn, sentence-transformers.
-  Para recuperación vectorial se debe usar ChromaDB o FAISS.
-  No se permite el uso de motores completos de b ́usqueda como Elasticsearch, Solr o Whoosh.
-  La implementación de los modelos clásicos debe ser propia o claramente explicada.
## Entregables
-  URL del despliegue del sistema.
-  Código fuente con instrucciones claras de ejecución (README).
-  Informe técnico (máximo 5 páginas), que incluya:
-  Descripción del corpus utilizado.
-  Explicación de las decisiones de diseño.
-  Ejemplos de consultas y resultados.
-  Análisis de métricas de evaluación.


## Veredicto Actual 15/11/2026

**El proyecto cubre bien el núcleo de recuperación y evaluación, pero todavía no cumple el documento completo.** La brecha principal es que no hay una interfaz web ni despliegue; también faltan instrucciones de ejecución y el informe comparativo solicitado.

| Requisito | Estado | Evidencia |
|---|---|---|
| Preprocesamiento e índice invertido | **Cumple** | Normaliza, elimina stopwords, aplica stemming y guarda frecuencias por término en `preprocessing.py` y `inverted_index.py`. |
| Jaccard, TF-IDF y BM25 | **Cumple** | Los tres modelos están implementados en `jaccard.py`, `tfidf.py` y `bm25.py`. |
| Embeddings y búsqueda vectorial | **Cumple** | Usa un modelo preentrenado y FAISS, tecnología permitida por el documento; véanse `embedder.py` y `vector_store.py`. |
| Consultas y rankings | **Cumple en consola** | Los modelos aceptan consultas libres y devuelven rankings. No encontré una interfaz para usarlos. |
| Evaluación con qrels, precisión, recall y MAP | **Cumple en el backend** | Se evalúan 648 consultas de prueba. Los resultados guardados en `metrics.json` muestran MAP@100: Jaccard **0,038**, TF-IDF **0,125**, BM25 **0,203** y embeddings **0,310**. |
| Interfaz web, comparación visual y despliegue | **No cumple / no hay evidencia en el repositorio** | No encontré código de interfaz, servidor web ni configuración de despliegue. `main.py` solo imprime un saludo. |
| Análisis por tipos de consulta e informe técnico | **Parcial / pendiente** | Hay métricas agregadas y por consulta, pero no el análisis de cuándo mejora o empeora la búsqueda semántica, ni un informe técnico de máximo cinco páginas. |
| README con instrucciones claras y URL | **Pendiente** | El `README.md` describe los requisitos, pero no cómo ejecutar el sistema ni incluye una URL de despliegue. |

Ejecuté las pruebas existentes de modelos, métricas y FAISS: **las tres pasaron**. En resumen, la base técnica está avanzada y los resultados guardados favorecen a embeddings en el conjunto de prueba; para ajustarse al documento, lo prioritario es desarrollar la interfaz y desplegarla, completar las instrucciones del README y preparar el informe comparativo.

## Pasos Usados hasta Ahora:
### Preparacion del Entorno.
Python en Entorno

```powershell
deactivate
Remove-Item -Recurse -Force .venv
uv python install 3.12
uv venv --python 3.12
.venv\Scripts\activate
```
Descarga de FiQA real en el entorno, el script busca los archivos por nombre y soporta parquet, jsonl y tsv por si el formato del repo difiere de lo esperado.

```Powershell
uv sync
uv run python -m backend.scripts.download_fiqa
uv run python -m backend.scripts.corpus_stats
```

### Preprocesamiento y el índice invertido.
```Powershell
uv run python -m backend.tests.test_models
uv run python -m backend.scripts.build_index
uv run python -m backend.scripts.try_models
```
Rendimiento: con un corpus sintético del mismo tamaño que FiQA (57k documentos), construir el índice tomó unos segundos y cada consulta unos 2 ms. Con FiQA real el tiempo de construcción será mayor por el stemming, pero no debería ser un problema. El archivo del índice pesó unos 12 MB en esa prueba; el real puede variar.

Cuando corras build_index y try_models, pásame la salida. Con eso pasamos al Día 3 en adelante: la evaluación (Precision, Recall y MAP) sobre los qrels de test, para ver cómo se comportan estos tres modelos antes de agregar los embeddings.


### Evaluación (Precision, Recall y MAP).
```Powershell
uv run python -m backend.tests.test_evaluation
uv run python -m backend.scripts.run_eval
```
> [!IMPORTANT] evaluation.py: 
Precision@k y Recall@k por consulta (k = 5, 10 y 20), AP por consulta y MAP@100 para todo el sistema. Recibe cualquier función search(query, k), así que los embeddings se conectarán sin cambios.

> [!IMPORTANT] test_evaluation.py: 
verifica las métricas contra un ejemplo calculado a mano.
run_eval.py: evalúa Jaccard, TF-IDF y BM25 sobre las 648 consultas de test, imprime una tabla comparativa y guarda backend/artifacts/metrics.json (con el detalle por consulta) para la web.

### Embeddings con FAISS.
```Powershell
uv run python -m backend.tests.test_vector_store
uv run python -m backend.scripts.build_embeddings --limit 2000   # prueba rápida
uv run python -m backend.scripts.build_embeddings                # corpus completo
uv run python -m backend.scripts.run_eval
```
Decisiones de diseño:

* **Modelo:** all-MiniLM-L6-v2 (384 dimensiones) con fastembed, que usa ONNX en lugar de torch. Es más liviano para Render y no necesita GPU. El modelo se cambia con --model.
* **Vector DB:** FAISS con producto interno sobre vectores normalizados (equivale a coseno). Guardo el índice en fp16, que ocupa unos 44 MB frente a unos 88 MB del índice plano. Así cabe en GitHub sin problemas.
* **Textos largos:** se truncan a 256 tokens. Se puede cambiar con --max-length.
Reanudable: el cálculo guarda lotes en backend/data/emb_chunks/, así que si se corta no pierdes el avance.

```Powershell
uv run python -m backend.tests.test_class_examples
uv run python -m backend.scripts.tune
uv run python -m backend.scripts.build_embeddings --limit 2000   # prueba rápida
uv run python -m backend.scripts.build_embeddings                # corpus completo
uv run python -m backend.scripts.run_eval
uv run python -m backend.scripts.compare_models --a embeddings --b bm25
```

> [!CAUTION] No comparar scores crudos entre modelos. Las clases de BM25 y embeddings insisten en que BM25 y coseno tienen escalas distintas. En la vista de comparación de la web conviene mostrar posiciones y solapamiento de IDs, y avisar que los scores no son comparables. Los qrels de FiQA son incompletos. Un documento sin juicio no es necesariamente no relevante (pooling). Un modelo semántico puede recuperar documentos útiles que los qrels no marcan, y eso castiga sus métricas. Vale la pena mencionarlo en el análisis.

