import argparse

from lib.keyword_search import (
    command_search, 
    command_build, 
    command_tf,
    command_idf,
)


def main() -> None:

    # ======== Parse args ========
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("id", type=int, help="Document ID for term frequency")
    tf_parser.add_argument("term", type=str, help="Term to query term frequency")

    idf_parser = subparsers.add_parser("idf", help="Get inverted document frequency")
    idf_parser.add_argument("term", type=str, help="Term to query term frequency")

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
                print(f"{i}. ({res['id']}) {res['title']}")

        case "tf":
            print("Term frequency")
            count = command_tf(args.id, args.term)
            print(f"(ID: {args.id}) {args.term}: {count}")

        case "idf":
            print("Inverted Document Frequency")
            idf = command_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

            pass

        case "build":
            print("Building inverted index...")
            command_build()
            print("Build successful!")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
