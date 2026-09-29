from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = "data/corpus.txt"

MAX_STORIES = 500

def clean_text(text: str) -> str:

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()


def build_corpus():

    print("Downloading TinyStories dataset...")

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train"
    )

    print(
        f"Available stories: {len(dataset):,}"
    )

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for row in dataset:

            if (
                MAX_STORIES is not None
                and count >= MAX_STORIES
            ):
                break

            story = clean_text(
                row.get("text", "")
            )

            if not story:
                continue

            file.write(story)

            file.write("\n\n")

            count += 1

            if count % 500 == 0:
                print(
                    f"Processed {count:,} stories"
                )

    print()

    print(
        f"Corpus created from "
        f"{count:,} stories."
    )

    print(
        f"Saved to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    build_corpus()