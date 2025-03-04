import os
from config.root import ROOT_DIR

# SYMSPELL
SYMSPELL_DICTIONARIES_PATHS = {
    "en": os.path.join(ROOT_DIR, "data", "dictionaries", "en-80k.txt"),
    "es": os.path.join(ROOT_DIR, "data", "dictionaries", "es-100l.txt")
}

# -- SPACY
SPACY_NAME_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm"
}
SPACY_DEFAULT_LANG_MODEL = "es"

# -- WHISPER
WHISPER_DEVICE = "cpu" # "cpu" | "cuda"
WHISPER_MODEL_VERSION = "small" # tiny | base | small | medium | large