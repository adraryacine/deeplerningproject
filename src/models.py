from __future__ import annotations

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers


def compile_regression_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def build_mlp_regressor(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="mlp_input")
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, name="aqi_prediction")(x)
    return compile_regression_model(keras.Model(inputs, outputs, name="mlp_regressor"), learning_rate=learning_rate)


def build_lstm_regressor(input_shape: Tuple[int, int], learning_rate: float = 1e-3) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="lstm_input")
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(1, name="aqi_prediction")(x)
    return compile_regression_model(keras.Model(inputs, outputs, name="lstm_regressor"), learning_rate=learning_rate)


def build_gru_regressor(input_shape: Tuple[int, int], learning_rate: float = 1e-3) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="gru_input")
    x = layers.GRU(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.GRU(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(1, name="aqi_prediction")(x)
    return compile_regression_model(keras.Model(inputs, outputs, name="gru_regressor"), learning_rate=learning_rate)
