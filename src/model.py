import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization
)


def build_model(
    vocabulary_size: int,
    sequence_length: int,
    embedding_dim: int,
    lstm_units_1: int,
    lstm_units_2: int,
    dense_units: int,
    dropout_rate: float,
    learning_rate: float
):

    model = Sequential([

        Input(
            shape=(sequence_length,)
        ),

        Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim
        ),

        SpatialDropout1D(
            0.15
        ),

        LSTM(
            lstm_units_1,
            return_sequences=True,
            dropout=0.15
        ),

        LayerNormalization(),

        LSTM(
            lstm_units_2,
            dropout=0.15
        ),

        LayerNormalization(),

        Dense(
            dense_units,
            activation="relu"
        ),

        Dropout(
            dropout_rate
        ),

        Dense(
            vocabulary_size,
            activation="softmax"
        )
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model