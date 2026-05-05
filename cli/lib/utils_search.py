import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH_DATA = os.path.join(PROJECT_ROOT, "data", "movies.json")
PATH_FILTER = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies() -> dict:
    with open(PATH_DATA, "r") as file:
        data = json.load(file)

    return data["movies"]

def load_stopwords() -> list[str]:
    with open(PATH_FILTER, "r") as file:
        data = file.read()

    data = data.splitlines()
    return data
