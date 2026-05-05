import argparse
import json
import string


def cull_non_alphanumerics(input_str: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    input_str = input_str.translate(table)
    return input_str


def tokenize(input_str: str) -> list[str]:
    tokens = [s.lower() for s in input_str.split() if s != ""]
    return tokens


def get_matches_in_lists(list1: list[str], list2: list[str]) -> list[str]:
    """Checking if a word in list2 is in list1 means that the tokens in list1 are also in list2."""
    matches = []
    for token in list1:
        match = list(filter(lambda ls: ls.startswith(token), list2))
        matches.extend(match)

    return matches


def search_keyword(kw: str, data_dict: dict, field: str = "title") -> list:
    matches = []
    for i in range(0, len(data_dict["movies"])):
        title: str = data_dict["movies"][i][field]
        kw_to_match: list[str] = tokenize(cull_non_alphanumerics(kw))
        data_to_match: list[str] = tokenize(cull_non_alphanumerics(title))

        if get_matches_in_lists(kw_to_match, data_to_match):
            matches.append(title)

    return matches


def format_search_output(list_results: list, max_results: int = 0) -> str:
    output: str = ""
    if not list_results:
        return "No results!"

    if max_results == 0:
        max_results = len(list_results)

    if max_results > len(list_results):
        max_results = len(list_results)

    for i in range(0, max_results):
        output += f"\n{i + 1}. {list_results[i]}"

    return output


def main() -> None:

    # ======== Print configs ========

    max_results = 5

    # ======== Input ========
    with open("data/movies.json", "r") as file:
        data_movies = json.load(file)

    # ======== Parse args ========
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # ======== Search ========

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results_search = search_keyword(args.query, data_movies)
            print(format_search_output(results_search, max_results))

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
