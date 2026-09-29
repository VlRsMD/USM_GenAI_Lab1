import argparse

from src.generator import TextGenerator


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate text using a "
            "trained LSTM model."
        )
    )

    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help=(
            "Initial text used to "
            "start generation."
        )
    )

    parser.add_argument(
        "--words",
        type=int,
        default=30,
        help=(
            "Number of new words "
            "to generate."
        )
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help=(
            "Sampling temperature. "
            "Lower values are more "
            "predictable."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of most likely words considered during generation."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    generator = TextGenerator()

    result = generator.generate(
        seed_text=args.seed,
        num_words=args.words,
        temperature=args.temperature,
        top_k = args.top_k
    )

    print()
    print("Generated text:")
    print("----------------")
    print(result)


if __name__ == "__main__":
    main()