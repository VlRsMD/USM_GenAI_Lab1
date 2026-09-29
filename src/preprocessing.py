from pathlib import Path
import re

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer

from config import VOCAB_SIZE


def load_corpus(path: str) -> str:

    corpus_path = Path(path)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: "
            f"{corpus_path.resolve()}"
        )

    text = corpus_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            "Corpus is empty."
        )

    return text


def normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"([.,!?;:])",
        r" \1 ",
        text
    )

    text = re.sub(
        r"[^a-z0-9.,!?;:'\- ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def build_tokenizer(
    text: str
) -> Tokenizer:

    text = normalize_text(text)

    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<UNK>",
        filters=""
    )

    tokenizer.fit_on_texts(
        [text]
    )

    return tokenizer


def create_training_sequences(
    text: str,
    tokenizer: Tokenizer,
    sequence_length: int
):

    text = normalize_text(text)

    tokens = tokenizer.texts_to_sequences(
        [text]
    )[0]

    if len(tokens) <= sequence_length:
        raise ValueError(
            "Corpus is too short."
        )

    print(
        f"Total tokens: {len(tokens):,}"
    )

    sequences = []

    for i in range(
        sequence_length,
        len(tokens)
    ):

        sequence = tokens[
            i - sequence_length:
            i + 1
        ]

        sequences.append(
            sequence
        )

    sequences = np.asarray(
        sequences,
        dtype=np.int32
    )

    x = sequences[:, :-1]
    y = sequences[:, -1]

    return x, y


def save_tokenizer(
    tokenizer: Tokenizer,
    path: str
):

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        tokenizer.to_json(),
        encoding="utf-8"
    )


def load_tokenizer(
    path: str
):

    from tensorflow.keras.preprocessing.text import (
        tokenizer_from_json
    )

    tokenizer_path = Path(path)

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: "
            f"{tokenizer_path.resolve()}"
        )

    tokenizer_json = (
        tokenizer_path.read_text(
            encoding="utf-8"
        )
    )

    return tokenizer_from_json(
        tokenizer_json
    )