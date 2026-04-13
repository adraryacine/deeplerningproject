from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_learning_curves(history, model_name: str, output_dir: Path) -> None:
    history_df = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_df["loss"], label="Train Loss")
    axes[0].plot(history_df["val_loss"], label="Validation Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    metric_column = "rmse" if "rmse" in history_df.columns else "mae"
    val_metric_column = f"val_{metric_column}"
    axes[1].plot(history_df[metric_column], label=f"Train {metric_column.upper()}")
    axes[1].plot(history_df[val_metric_column], label=f"Validation {metric_column.upper()}")
    axes[1].set_title(f"{model_name} - {metric_column.upper()}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_column.upper())
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prediction_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    errors = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--")
    axes[0].set_title(f"{model_name} - Valeurs reelles vs predictions")
    axes[0].set_xlabel("AQI reel")
    axes[0].set_ylabel("AQI predit")

    axes[1].hist(errors, bins=40, color="teal", alpha=0.8)
    axes[1].set_title(f"{model_name} - Distribution des erreurs")
    axes[1].set_xlabel("Erreur (reel - predit)")
    axes[1].set_ylabel("Frequence")

    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if len(dates) > 0:
        series_dates = pd.to_datetime(pd.Series(dates), errors="coerce")
        order = np.argsort(series_dates.values.astype("datetime64[ns]"))
        sample_count = min(300, len(order))
        sampled_idx = order[-sample_count:]

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(series_dates.iloc[sampled_idx], y_true[sampled_idx], label="Reel", linewidth=2)
        ax.plot(series_dates.iloc[sampled_idx], y_pred[sampled_idx], label="Predit", linewidth=2)
        ax.set_title(f"{model_name} - Serie temporelle sur le test")
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / f"{model_name}_timeseries_test.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def permutation_importance_sequence_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    baseline_pred = model.predict(X_test, verbose=0).reshape(-1)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    n_base_features = len(feature_names)
    rows = []

    for feature_idx, feature_name in enumerate(feature_names):
        X_permuted = X_test.copy()

        if X_permuted.ndim == 3:
            shuffled = X_permuted[:, :, feature_idx].copy()
            rng.shuffle(shuffled, axis=0)
            X_permuted[:, :, feature_idx] = shuffled
        else:
            feature_positions = np.arange(feature_idx, X_permuted.shape[1], n_base_features)
            shuffled = X_permuted[:, feature_positions].copy()
            rng.shuffle(shuffled, axis=0)
            X_permuted[:, feature_positions] = shuffled

        permuted_pred = model.predict(X_permuted, verbose=0).reshape(-1)
        permuted_rmse = np.sqrt(mean_squared_error(y_test, permuted_pred))
        rows.append({"feature": feature_name, "rmse_increase": float(permuted_rmse - baseline_rmse)})

    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(importance_df: pd.DataFrame, output_path: Path, top_k: int = 15) -> None:
    top_df = importance_df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"], top_df["rmse_increase"], color="darkorange")
    ax.set_title("Importance par permutation")
    ax.set_xlabel("Hausse de RMSE apres permutation")
    ax.set_ylabel("Variable")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
