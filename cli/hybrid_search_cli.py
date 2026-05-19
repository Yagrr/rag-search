import argparse

from lib.hybrid_search import command_normalize

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize list of input scores using Min-Max method")
    normalize_parser = normalize_parser.add_argument("values", type=float, nargs="*", help="List of values separated by spaces")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            command_normalize(args.values)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
