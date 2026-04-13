from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import pandas as pd
from tensorflow import keras

from .data_loader import infer_schema, inspect_dataset, load_dataset
from .evaluate import (
    permutation_importance_sequence_model,
    regression_metrics,
    save_feature_importance_plot,
    save_learning_curves,
    save_prediction_plots,
)
from .models import build_gru_regressor, build_lstm_regressor, build_mlp_regressor
from .preprocessing import (
    clean_air_quality_data,
    fit_transform_datasets,
    save_preprocessor,
    temporal_train_val_test_split,
)
from .sequence_builder import build_sequences
from .utils import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories, save_json, set_global_seed


@dataclass
class TrainingConfig:
    dataset_path: str | None = None
    sequence_length: int = 14
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 1e-3
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


def _callbacks(model_name: str) -> list:
    checkpoint_path = MODELS_DIR / f"{model_name}_best.keras"
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]


def _train_single_model(model, model_name, X_train, y_train, X_val, y_val, batch_size, epochs):
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=_callbacks(model_name),
    )


def run_training(config: TrainingConfig) -> Dict:
    ensure_directories()
    set_global_seed(config.random_seed)

    raw_df = load_dataset(config.dataset_path)
    schema = infer_schema(raw_df, config.dataset_path)
    inspection = inspect_dataset(raw_df, schema)

    cleaned_df = clean_air_quality_data(raw_df, schema)
    train_df, val_df, test_df = temporal_train_val_test_split(
        cleaned_df,
        date_column=schema.date_column,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    artifacts = fit_transform_datasets(train_df, val_df, test_df, schema)
    save_preprocessor(artifacts.preprocessor, str(MODELS_DIR / "preprocessor.joblib"))

    train_seq = build_sequences(
        artifacts.train_frame,
        artifacts.X_train,
        artifacts.y_train,
        schema.date_column,
        schema.group_column,
        config.sequence_length,
    )
    val_seq = build_sequences(
        artifacts.val_frame,
        artifacts.X_val,
        artifacts.y_val,
        schema.date_column,
        schema.group_column,
        config.sequence_length,
    )
    test_seq = build_sequences(
        artifacts.test_frame,
        artifacts.X_test,
        artifacts.y_test,
        schema.date_column,
        schema.group_column,
        config.sequence_length,
    )

    model_registry = {
        "mlp": build_mlp_regressor(train_seq.X_flat.shape[1], learning_rate=config.learning_rate),
        "lstm": build_lstm_regressor(
            (train_seq.X_seq.shape[1], train_seq.X_seq.shape[2]),
            learning_rate=config.learning_rate,
        ),
        "gru": build_gru_regressor(
            (train_seq.X_seq.shape[1], train_seq.X_seq.shape[2]),
            learning_rate=config.learning_rate,
        ),
    }
    model_inputs = {
        "mlp": (train_seq.X_flat, val_seq.X_flat, test_seq.X_flat),
        "lstm": (train_seq.X_seq, val_seq.X_seq, test_seq.X_seq),
        "gru": (train_seq.X_seq, val_seq.X_seq, test_seq.X_seq),
    }

    results = []
    best_model_name = None
    best_rmse = float("inf")

    for model_name, model in model_registry.items():
        X_train, X_val, X_test = model_inputs[model_name]
        history = _train_single_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            y_train=train_seq.y,
            X_val=X_val,
            y_val=val_seq.y,
            batch_size=config.batch_size,
            epochs=config.epochs,
        )

        predictions = model.predict(X_test, verbose=0).reshape(-1)
        metrics = regression_metrics(test_seq.y, predictions)
        results.append({"model": model_name.upper(), **metrics})

        save_learning_curves(history, model_name, FIGURES_DIR)
        save_prediction_plots(test_seq.y, predictions, test_seq.dates, model_name, FIGURES_DIR)

        pd.DataFrame(
            {
                "date": test_seq.dates,
                "group": test_seq.groups,
                "y_true": test_seq.y,
                "y_pred": predictions,
                "error": test_seq.y - predictions,
            }
        ).to_csv(REPORTS_DIR / f"{model_name}_test_predictions.csv", index=False)

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model_name = model_name

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_model = keras.models.load_model(MODELS_DIR / f"{best_model_name}_best.keras")
    best_test_input = model_inputs[best_model_name][2]
    best_importance = permutation_importance_sequence_model(
        best_model,
        best_test_input,
        test_seq.y,
        artifacts.feature_names,
    )
    best_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    save_feature_importance_plot(best_importance, FIGURES_DIR / "feature_importance.png")

    run_summary = {
        "config": asdict(config),
        "schema": {
            "date_column": schema.date_column,
            "group_column": schema.group_column,
            "target_column": schema.target_column,
            "task_type": schema.task_type,
            "classification_target": schema.classification_target,
            "numeric_features": schema.numeric_features,
            "categorical_features": schema.categorical_features,
            "pollutant_columns": schema.pollutant_columns,
            "target_reason": schema.target_reason,
        },
        "inspection": inspection,
        "cleaned_shape": list(cleaned_df.shape),
        "split_shapes": {
            "train": list(train_df.shape),
            "validation": list(val_df.shape),
            "test": list(test_df.shape),
        },
        "sequence_shapes": {
            "train": list(train_seq.X_seq.shape),
            "validation": list(val_seq.X_seq.shape),
            "test": list(test_seq.X_seq.shape),
        },
        "results": results_df.to_dict(orient="records"),
        "best_model": best_model_name,
    }
    save_json(run_summary, REPORTS_DIR / "run_summary.json")
    return run_summary
