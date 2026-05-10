#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, verify_embedding, embed_text

def main():

    # ======== Parse args ========
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    model_parser = subparsers.add_parser("verify", help="Verify if model is loaded")
    model_parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Search query")

    subparsers.add_parser("verify_embeddings", help="Verify if embeddings are available")

    embed_parser = subparsers.add_parser("embed_text", help="Embed text to vector space using an embedding model (default: all-MiniLM-L6-v2)")
    embed_parser.add_argument("text", type=str, help="Text to embed")


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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
