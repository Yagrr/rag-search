#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    verify_embedding,
    embed_text,
    embed_query_text,
    command_semantic_search,
    chunk_text,
    chunk_text_semantically,
    embed_chunks,
    command_chunked_semantic_search,
)

from lib.utils_search import (
    DEFAULT_MODEL,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_WORDS_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_OVERLAP,
)


def main():

    # ======== Parse args ========
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    model_parser = subparsers.add_parser("verify", help="Verify if model is loaded")
    model_parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Search query")

    subparsers.add_parser("verify_embeddings", help="Verify if embeddings are available")

    single_embed_parser = subparsers.add_parser("embed_text", help="Embed text to vector space using an embedding model (default: all-MiniLM-L6-v2)")
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query to vector space using an embedding model (default: all-MiniLM-L6-v2)")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Semantically search movies")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit search results")

    chunk_parser = subparsers.add_parser("chunk", help="Split long text data into smaller fixed-size pieces for embedding")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Set default word limit in each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_WORDS_OVERLAP, help="Set number of overlapping words to preserve context across chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split long text data semantically into smaller context-aware pieces for embedding")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to semantically chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_SEMANTIC_CHUNK_SIZE, help="Set maximum word limit in each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_SEMANTIC_CHUNK_OVERLAP, help="Set number of overlapping words to preserve further context across semantically-split chunks")

    subparsers.add_parser("embed_chunks", help="Generate semantic chunks")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Perform a chunked semantic search")
    search_chunked_parser.add_argument("query", type=str, help="Query to search")
    search_chunked_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit search results")

    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying model...")
            verify_model(args.model)

        case "verify_embeddings":
            print("Verifying embeddings...")
            verify_embedding()

        case "embed_text":
            print("Embedding text...")
            embed_text(args.text)

        case "embed_query":
            print("Embedding query...")
            embed_query_text(args.query)

        case "chunk":
            print("Chunking text...")
            chunk_text(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            print("Semantically chunking text...")
            chunk_text_semantically(args.text, args.max_chunk_size, args.overlap)

        case "search":
            print("Starting semantic search...")
            command_semantic_search(args.query, args.limit)

        case "embed_chunks":
            print("Embedding chunks...")
            embeddings = embed_chunks()
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            print("Starting chunked semantic search...")
            command_chunked_semantic_search(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
