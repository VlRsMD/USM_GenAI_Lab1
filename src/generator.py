import re

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

from config import (
    MODEL_PATH,
    SEQUENCE_LENGTH,
    TOKENIZER_PATH,
)

from src.preprocessing import (
    load_tokenizer,
    normalize_text,
)


class TextGenerator:

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        tokenizer_path: str = TOKENIZER_PATH,
        sequence_length: int = SEQUENCE_LENGTH,
    ):

        self.model = tf.keras.models.load_model(
            model_path
        )

        self.tokenizer = load_tokenizer(
            tokenizer_path
        )

        self.sequence_length = sequence_length

        self.index_to_word = {
            index: word
            for word, index
            in self.tokenizer.word_index.items()
        }

        self.unk_id = (
            self.tokenizer.word_index.get("<UNK>")
            or self.tokenizer.word_index.get("<unk>")
        )

    @staticmethod
    def sample_top_k(
        probabilities,
        temperature: float = 0.6,
        k: int = 8,
    ) -> int:

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than 0."
            )

        if k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        ).copy()

        probabilities[0] = 0.0

        valid_count = np.count_nonzero(
            probabilities > 0
        )

        if valid_count == 0:
            raise ValueError(
                "No valid token probabilities available."
            )

        k = min(
            k,
            valid_count,
        )

        top_indices = np.argpartition(
            probabilities,
            -k,
        )[-k:]

        top_probabilities = probabilities[
            top_indices
        ]

        logits = np.log(
            top_probabilities + 1e-10
        )

        logits /= temperature

        logits -= np.max(logits)

        top_probabilities = np.exp(
            logits
        )

        probability_sum = (
            top_probabilities.sum()
        )

        if probability_sum <= 0:
            return int(
                top_indices[
                    np.argmax(
                        probabilities[top_indices]
                    )
                ]
            )

        top_probabilities /= probability_sum

        selected_index = np.random.choice(
            len(top_indices),
            p=top_probabilities,
        )

        return int(
            top_indices[selected_index]
        )

    def apply_repetition_penalty(
        self,
        probabilities,
        generated_tokens,
        penalty: float = 0.4,
        window_size: int = 6,
    ):

        recent_tokens = generated_tokens[
            -window_size:
        ]

        for token_id in recent_tokens:

            word = self.index_to_word.get(
                token_id
            )

            if word is None:
                continue

            # Do not penalize punctuation.
            if word in {
                ".",
                ",",
                "!",
                "?",
                ";",
                ":",
            }:
                continue

            probabilities[token_id] *= penalty

        return probabilities

    @staticmethod
    def format_generated_text(
        tokens
    ) -> str:

        text = " ".join(tokens)

        text = re.sub(
            r"\s+([.,!?;:])",
            r"\1",
            text,
        )

        text = re.sub(
            r"([.,!?;:])([A-Za-z0-9])",
            r"\1 \2",
            text,
        )

        if text:
            text = (
                text[0].upper()
                + text[1:]
            )

        return text

    def generate(
        self,
        seed_text: str,
        num_words: int = 30,
        temperature: float = 0.6,
        top_k: int = 8,
        repetition_penalty: float = 0.4,
    ) -> str:

        if not seed_text.strip():
            raise ValueError(
                "Seed text cannot be empty."
            )

        if num_words <= 0:
            raise ValueError(
                "num_words must be greater than 0."
            )

        normalized_seed = normalize_text(
            seed_text
        )

        generated_tokens = (
            self.tokenizer.texts_to_sequences(
                [normalized_seed]
            )[0]
        )

        if not generated_tokens:
            raise ValueError(
                "Seed text contains no recognized tokens."
            )

        original_seed_length = len(
            generated_tokens
        )

        generated_count = 0

        while generated_count < num_words:

            input_tokens = generated_tokens[
                -self.sequence_length:
            ]

            input_tokens = pad_sequences(
                [input_tokens],
                maxlen=self.sequence_length,
                padding="pre",
                truncating="pre",
            )

            predictions = self.model.predict(
                input_tokens,
                verbose=0,
            )[0].astype(
                np.float64
            )

            predictions[0] = 0.0

            if self.unk_id is not None:
                predictions[
                    self.unk_id
                ] = 0.0

            predictions = (
                self.apply_repetition_penalty(
                    probabilities=predictions,
                    generated_tokens=generated_tokens,
                    penalty=repetition_penalty,
                )
            )

            total = predictions.sum()

            if total <= 0:
                raise RuntimeError(
                    "Model produced no valid "
                    "token probabilities."
                )

            predictions /= total

            predicted_id = (
                self.sample_top_k(
                    probabilities=predictions,
                    temperature=temperature,
                    k=top_k,
                )
            )

            predicted_word = (
                self.index_to_word.get(
                    predicted_id
                )
            )

            if predicted_word is None:
                continue

            if predicted_word.lower() == "<unk>":
                continue

            generated_tokens.append(
                predicted_id
            )

            generated_count += 1

        words = []

        for token_id in generated_tokens:

            word = self.index_to_word.get(
                token_id
            )

            if word is not None:
                words.append(word)

        return self.format_generated_text(
            words
        )