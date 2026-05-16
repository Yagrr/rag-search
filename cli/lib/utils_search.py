import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH_DATA = os.path.join(PROJECT_ROOT, "data", "movies.json")
PATH_FILTER = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
PATH_CACHE = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_WORDS_OVERLAP = 2

DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_SEMANTIC_CHUNK_OVERLAP = 1
SCORE_PRECISION = 6

# controls TF value saturation
BM25_K1 = 1.5
# controls saturation effect in length normalization
BM25_B = 0.75

def load_movies() -> dict:
    with open(PATH_DATA, "r") as file:
        data = json.load(file)

    return data["movies"]


def load_stopwords() -> list[str]:
    with open(PATH_FILTER, "r") as file:
        data = file.read()

    data = data.splitlines()
    return data
