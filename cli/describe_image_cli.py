import argparse
from lib.describe_image import command_describe_image

def main() -> None:
    parser = argparse.ArgumentParser(description="Image describer CLI")
    parser.add_argument("--image", required=True, type=str, help="Path to image file")
    parser.add_argument("--query", required=True, type=str, help="Query to rewrite based on image")

    args = parser.parse_args()

    if not args.image or not args.query:
        parser.print_help()
    else:
        command_describe_image(args.image, args.query)

if __name__ == "__main__":
    main()
