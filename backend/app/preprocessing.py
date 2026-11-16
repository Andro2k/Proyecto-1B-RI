"""Pipeline de preprocesamiento compartido por documentos y consultas:
tokenización (regex) -> minúsculas -> remoción de stopwords -> stemming (Snowball inglés, "Porter2").

Se usa py-rust-stemmers en vez de NLTK: solo importar nltk.stem consume ~150 MB de RAM y el plan
gratuito de Render tiene 512 MB. Ambos stemmers coinciden en ~98 % de las palabras."""
import re
from functools import lru_cache
from pathlib import Path

from py_rust_stemmers import SnowballStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    Path(__file__).with_name("stopwords_en.txt").read_text(encoding="utf-8").split()
)
_stemmer = SnowballStemmer("english")


@lru_cache(maxsize=None)
def _stem(token: str) -> str:
    return _stemmer.stem_word(token)


def preprocess(text: str) -> list[str]:
    """Devuelve la lista de términos (con repeticiones) de un texto."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS]
