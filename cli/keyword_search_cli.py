import argparse

from lib.keyword_search import (
    command_search, 
    command_build, 
    command_tf,
    command_idf,
    command_tfidf,
    command_bm25_search,
    command_bm25_tf,
    command_bm25_idf,
)

from lib.utils_search import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT


def main() -> None:

    # ======== Parse args ========
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Search query")

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

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document and term")
    bm25tf_parser.add_argument("id", type=int, help="Document ID for BM25 term frequency (BM25 TF)")
    bm25tf_parser.add_argument("term", type=str, help="Term to query BM25 term frequency")
    bm25tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Search query")

    args = parser.parse_args()

    # ======== Search ========

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            docs_matching = command_search(args.query)
            if not docs_matching:
                print("No results found.")
            for i, doc in enumerate(docs_matching, 1):
                print(f"{i}. ({doc['id']}) {doc['title']}")

        case "tf":
            count = command_tf(args.id, args.term)
            print(f"Term frequency of '{args.term}' in (ID: {args.id}): {count}")

        case "idf":
            idf = command_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tfidf = command_tfidf(args.id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tfidf:.2f}'")

        case "bm25search":
            print("BM25 Search:")
            docs_bm25_matching, scores_bm25 = command_bm25_search(args.query, args.limit)
            if not scores_bm25:
                print("No results found.")
            for i, doc in enumerate(docs_bm25_matching, 1):
                id = doc['id']
                print(f"{i}. ({id}) {doc['title']} - Score: {scores_bm25[id]:.2f}")

        case "bm25tf":
            bm25_tf = command_bm25_tf(args.id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.id}': {bm25_tf:.2f}")

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
