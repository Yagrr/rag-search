import argparse

from lib.utils_search import DEFAULT_SEARCH_LIMIT
from lib.multimodal_search import verify_image_embedding, command_image_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify if embedding can be generated from an input path to image")
    verify_embedding_parser.add_argument("path_image", type=str, help="Path to image")

    image_search_parser = subparsers.add_parser("image_search", help="Search with image")
    image_search_parser.add_argument("path_image", type=str, help="Path to image")
    image_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path_image)

        case "image_search":
            command_image_search(args.path_image, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
