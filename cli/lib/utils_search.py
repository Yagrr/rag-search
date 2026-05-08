import json
import os

DEFAULT_SEARCH_LIMIT = 5
# controls TF value saturation
BM25_K1 = 1.5
# controls saturation effect in length normalization
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH_DATA = os.path.join(PROJECT_ROOT, "data", "movies.json")
PATH_FILTER = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
PATH_CACHE = os.path.join(PROJECT_ROOT, "cache")


def load_movies() -> dict:
    with open(PATH_DATA, "r") as file:
        data = json.load(file)

    return data["movies"]


def load_stopwords() -> list[str]:
    with open(PATH_FILTER, "r") as file:
        data = file.read()

    data = data.splitlines()
    return data
