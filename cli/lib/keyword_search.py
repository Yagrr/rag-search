import string

from .utils_search import DEFAULT_SEARCH_LIMIT

# ======== Pre-processing  ========


def command_search(query: str, data: dict, field_to_search: str = "title", limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    matches = []

    for i, datum in enumerate(data):
        field_values: str = data[i][field_to_search]
        query_to_match: list[str] = tokenize_text(preprocess_text(query))
        data_to_match: list[str] = tokenize_text(preprocess_text(field_values))

        if get_matching_tokens(query_to_match, data_to_match):
            matches.append(datum)
            if len(matches) >= limit:
                break

    return matches


def preprocess_text(input_str: str) -> str:
    """ Cull non alphanumeric characters
    """
    input_str = input_str.lower()
    input_str = input_str.translate(str.maketrans("", "", string.punctuation))
    return input_str


def tokenize_text(input_str: str) -> list[str]:
    """ Tokenize text by whitespace inbetween words.
    Cull empty strings
    """
    tokens = [s for s in input_str.split() if s != ""]
    return tokens


def get_matching_tokens(list1: list[str], list2: list[str]) -> list[str]:
    """Checking if a word in list2 is in list1 means that the tokens in list1 are also in list2."""
    matching_tokens = []
    for token in list1:
        match = list(filter(lambda ls: ls.startswith(token), list2))
        matching_tokens.extend(match)

    return matching_tokens


def print_search_results(list_results: list, max_results: int = 0) -> None:
    if not list_results:
        return print("No results!")

    if max_results == 0:
        max_results = len(list_results)

    if max_results > len(list_results):
        max_results = len(list_results)

    for i in range(0, max_results):
        print(f"\n{i + 1}. {list_results[i]}")
