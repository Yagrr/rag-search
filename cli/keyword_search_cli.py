import argparse

from lib.keyword_search import command_search
from lib.utils_search import load_movies
from lib.tf_idf import InvertedIndex


def main() -> None:

    # ======== Parse args ========
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser = subparsers.add_parser("build", help="Build index")

    args = parser.parse_args()
    data = load_movies()

    # ======== Search ========

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = command_search(args.query, data)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")

        case "build":
            print("Building inverted index...")
            docs = InvertedIndex()
            docs.build()
            print("Build successful!")
            docs.save()
            print(f"First document for token 'merida' = {docs.get_documents('merida')}")



        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
