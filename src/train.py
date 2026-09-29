from pathlib import Path

import tensorflow as tf

from config import (
    BATCH_SIZE,
    CORPUS_PATH,
    DENSE_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    EPOCHS,
    LEARNING_RATE,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    TOKENIZER_PATH,
    VALIDATION_SPLIT,
)

from src.model import build_model

from src.preprocessing import (
    build_tokenizer,
    create_training_sequences,
    load_corpus,
    save_tokenizer,
)


def train():
    print("Loading corpus.txt...")

    text = load_corpus(
        CORPUS_PATH
    )

    print("Building tokenizer...")

    tokenizer = build_tokenizer(
        text
    )

    vocabulary_size = (
        len(tokenizer.word_index) + 1
    )

    print(
        f"Vocabulary size: "
        f"{vocabulary_size}"
    )

    print(
        "Creating training sequences..."
    )

    x, y = create_training_sequences(
        text=text,
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH
    )

    print(
        f"Training examples: {len(x)}"
    )

    print(
        f"Input shape: {x.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    model = build_model(
        vocabulary_size=vocabulary_size,
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        lstm_units_1=LSTM_UNITS_1,
        lstm_units_2=LSTM_UNITS_2,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )

    model.summary()

    Path(MODEL_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print("Starting training...")

    model.fit(
        x,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        shuffle=True
    )

    save_tokenizer(
        tokenizer,
        TOKENIZER_PATH
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Tokenizer saved to: "
        f"{TOKENIZER_PATH}"
    )


if __name__ == "__main__":
    train()