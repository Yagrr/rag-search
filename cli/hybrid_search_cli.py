import argparse

from lib.hybrid_search import (
    normalize,
    command_weighted_search,
    command_rrf_search,
)
from lib.utils_search import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_WEIGHTED_SEARCH_ALPHA,
    DEFAULT_RRF_SEARCH_K,
)

from lib.llm import (
    enhance_query,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize list of input scores using Min-Max method"
    )
    normalize_parser = normalize_parser.add_argument(
        "values", type=float, nargs="*", help="List of values separated by spaces"
    )

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform a hybrid search combining BM25 and semantic search with weighting between the two")
    weighted_search_parser.add_argument("query", help="Text to query")
    weighted_search_parser.add_argument("--alpha", type=float, default=DEFAULT_WEIGHTED_SEARCH_ALPHA, help="Alpha weighting between BM25/semantic search")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit number of search results")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform a hybrid search using Reciprocal Rank Fusion")
    rrf_search_parser.add_argument("query", help="Text to query")
    rrf_search_parser.add_argument("-k", type=int, default=DEFAULT_RRF_SEARCH_K)
    rrf_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method",)


    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            command_weighted_search(args.query, args.alpha, args.limit)

        case "rrf-search":
            if args.enhance is not None:
                query = enhance_query(args.query, args.enhance)
            else:
                query = args.query
            command_rrf_search(query, args.k, args.limit)

        case "normalize":
            scores_normalized = normalize(args.values)
            for score in scores_normalized:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
