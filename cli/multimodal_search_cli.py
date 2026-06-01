import argparse
from enum import verify

from lib.multimodal_search import verify_image_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    multimodal_parser = subparsers.add_parser("verify_image_embedding", help="Verify if embedding can be generated from an input path to image")
    multimodal_parser.add_argument("path_image", type=str, help="Path to image")

    args = parser.parse_args()

    if not args.path_image:
        parser.print_help()
    else:
        verify_image_embedding(args.path_image)

if __name__ == "__main__":
    main()
