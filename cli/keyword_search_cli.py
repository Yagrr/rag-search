import argparse

from lib.keyword_search import command_search, command_build


def main() -> None:

    # ======== Parse args ========
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser = subparsers.add_parser("build", help="Build index")

    args = parser.parse_args()

    # ======== Search ========

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = command_search(args.query)
            if not results:
                print("No results found")
            for i, res in enumerate(results, 1):
                # TODO: results only contain ID. Need to find a way to return full doc
                print(f"{i}. ({res['id']}) {res['title']}")

        case "build":
            print("Building inverted index...")
            command_build()
            print("Build successful!")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
