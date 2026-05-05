import argparse
import json


def search_keyword(kw: str, dict_movies: dict) -> list:
    matches = []

    for i in range(0, len(dict_movies["movies"])):
        title = dict_movies["movies"][i]["title"]
        if kw in title:
            matches.append(title)

    return matches


def format_search_output(list_results: list, max_results: int = 0) -> str:
    output: str = ""
    if not list_results:
        return "No results!"

    if max_results == 0:
        max_results = len(list_results)

    for i in range(0, max_results):
        output += f"\n{i + 1}. {list_results[i]}\n"

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
