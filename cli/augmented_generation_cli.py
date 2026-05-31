import argparse
from lib.utils_search import DEFAULT_SEARCH_LIMIT
from lib.augmented_generation import command_rag, command_summarize

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    rag_parser = subparsers.add_parser(
        "summarize", help="Synthesize search results"
    )
    rag_parser.add_argument("query", type=str, help="Query to summarizel")
    rag_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Query to summarizel")
    

    args = parser.parse_args()

    match args.command:
        case "rag":
            command_rag(args.query)

        case "summarize":
            command_summarize(args.query)
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
