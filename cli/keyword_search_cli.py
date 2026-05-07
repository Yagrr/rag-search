import argparse

from lib.keyword_search import (
    command_search, 
    command_build, 
    command_tf,
    command_idf,
    command_tfidf,
    command_bm25_idf,
)


def main() -> None:

    # ======== Parse args ========
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("id", type=int, help="Document ID for term frequency (TF)")
    tf_parser.add_argument("term", type=str, help="Term to query term frequency")

    idf_parser = subparsers.add_parser("idf", help="Get inverted document frequency (IDF)")
    idf_parser.add_argument("term", type=str, help="Term to query IDF score")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF")
    tfidf_parser.add_argument("id", type=int, help="Document ID for TF-IDF")
    tfidf_parser.add_argument("term", type=str, help="Term to obtain TF-IDF score")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF")
    bm25idf_parser.add_argument("term", type=str, help="Term to obtain BM25 IDF score")


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
            count = command_tf(args.id, args.term)
            print(f"Term frequency of '{args.term}' in (ID: {args.id}): {count}")

        case "idf":
            idf = command_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tfidf = command_tfidf(args.id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tfidf:.2f}'")

        case "bm25idf":
            bm25_idf = command_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")

        case "build":
            print("Building inverted index...")
            command_build()
            print("Build successful!")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
