import io
import re
from typing import Tuple
from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.pdfparser import PDFSyntaxError
import nltk

# Ensure necessary NLTK data is available (download quietly at runtime if missing)
def _safe_nltk_download(resource: str):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True)

_safe_nltk_download("tokenizers/punkt")
_safe_nltk_download("corpora/stopwords")
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

STOPWORDS_PT = set(stopwords.words("portuguese"))
STOPWORDS_EN = set(stopwords.words("english"))
STEMMER_PT = SnowballStemmer("portuguese")
STEMMER_EN = SnowballStemmer("english")

def extract_text_from_upload(stream, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        return stream.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        # pdfminer expects a file-like object. We already have stream.
        try:
            # Reset pointer in case
            stream.seek(0)
        except Exception:
            pass
        try:
            text = pdf_extract_text(stream)
        except PDFSyntaxError:
            text = ""
        return text or ""
    return ""

def preprocess_text(text: str) -> str:
    # Basic cleanup
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())

    # Language-agnostic filtering: remove stopwords (pt + en) & short tokens
    filtered = [t for t in tokens if t not in STOPWORDS_PT and t not in STOPWORDS_EN and len(t) > 2]

    # Light stemming (pt + en)
    stemmed = []
    for t in filtered:
        # Try portuguese stem, else english
        try:
            st = STEMMER_PT.stem(t)
        except Exception:
            st = t
        try:
            st_en = STEMMER_EN.stem(st)
        except Exception:
            st_en = st
        stemmed.append(st_en)

    return " ".join(stemmed)
