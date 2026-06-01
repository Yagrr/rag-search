import argparse
from lib.utils_search import DEFAULT_SEARCH_LIMIT
from lib.augmented_generation import (
    command_rag, 
    command_summarize,
    command_citations,
    command_question,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Synthesize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Query to summarize")
    summarize_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Query to summarizel")

    citations_parser = subparsers.add_parser(
        "citations", help="Synthesize search results"
    )
    citations_parser.add_argument("query", type=str, help="Query to summarize and add citations")
    citations_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Query to summarizel")

    question_parser = subparsers.add_parser(
        "question", help="Ask a question about the dataset"
    )
    question_parser.add_argument("query", type=str, help="Query to ask a question about the dataset")
    question_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Query to summarizel")

    args = parser.parse_args()

    match args.command:
        case "rag":
            command_rag(args.query)

        case "summarize":
            command_summarize(args.query, args.limit)

        case "citations":
            command_citations(args.query)

        case "question":
            command_question(args.query, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
