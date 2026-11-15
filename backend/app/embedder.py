"""Generación de embeddings con un modelo preentrenado (fastembed = ONNX, sin torch: liviano para Render).
Todos los vectores salen normalizados (norma 1), así el producto interno equivale al coseno."""
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensiones
DEFAULT_MAX_LENGTH = 256  # tokens; el p95 de FiQA son ~370 palabras, se trunca el final de los más largos


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class FastEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        from fastembed import TextEmbedding  # import perezoso: solo se necesita al embeber

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)
        # fastembed no expone max_length: se fija la truncación directamente en el tokenizador ONNX
        onnx_model = self.model.model
        if hasattr(onnx_model, "_ensure_tokenizer"):
            onnx_model._ensure_tokenizer()
        onnx_model.tokenizer.enable_truncation(max_length=max_length)

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return _normalize(np.array(list(self.model.passage_embed(texts, batch_size=batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self.model.query_embed(text))))  # forma (1, dim)
