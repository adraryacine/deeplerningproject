from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_loader import DatasetSchema


sns.set_theme(style="whitegrid", context="notebook")


def generate_eda_plots(df: pd.DataFrame, schema: DatasetSchema, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_missing_values(df, output_dir / "missing_values.png")
    _plot_target_distribution(df, schema.target_column, output_dir / "target_distribution.png")
    _plot_correlation_heatmap(df, output_dir / "correlation_heatmap.png")
    _plot_pollutant_timeseries(df, schema, output_dir / "pollutant_timeseries.png")
    _plot_monthly_trend(df, schema, output_dir / "monthly_aqi_trend.png")
    _plot_boxplots(df, schema, output_dir / "feature_boxplots.png")


def _plot_missing_values(df: pd.DataFrame, output_path: Path) -> None:
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index, palette="mako", ax=ax)
    ax.set_title("Taux de valeurs manquantes par variable")
    ax.set_xlabel("Proportion de valeurs manquantes")
    ax.set_ylabel("Variable")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_target_distribution(df: pd.DataFrame, target_column: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[target_column].dropna(), kde=True, bins=40, color="steelblue", ax=axes[0])
    axes[0].set_title(f"Distribution de {target_column}")
    axes[0].set_xlabel(target_column)
    sns.boxplot(x=df[target_column].dropna(), color="salmon", ax=axes[1])
    axes[1].set_title(f"Boxplot de {target_column}")
    axes[1].set_xlabel(target_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Heatmap de correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pollutant_timeseries(df: pd.DataFrame, schema: DatasetSchema, output_path: Path) -> None:
    if not schema.date_column or not schema.pollutant_columns:
        return
    available = [col for col in schema.pollutant_columns if col in df.columns][:5]
    if not available:
        return
    daily = df.groupby(schema.date_column)[available].mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(15, 6))
    for column in available:
        ax.plot(daily[schema.date_column], daily[column], label=column, linewidth=1.5)
    ax.set_title("Evolution temporelle moyenne des principaux polluants")
    ax.set_xlabel("Date")
    ax.set_ylabel("Concentration moyenne")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_monthly_trend(df: pd.DataFrame, schema: DatasetSchema, output_path: Path) -> None:
    if not schema.date_column or schema.target_column not in df.columns:
        return
    working_df = df.copy()
    working_df["month_period"] = working_df[schema.date_column].dt.to_period("M").astype(str)
    monthly = working_df.groupby("month_period")[schema.target_column].mean().reset_index()
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(monthly["month_period"], monthly[schema.target_column], color="purple", linewidth=2)
    ax.set_title(f"Tendance mensuelle moyenne de {schema.target_column}")
    ax.set_xlabel("Mois")
    ax.set_ylabel(schema.target_column)
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_boxplots(df: pd.DataFrame, schema: DatasetSchema, output_path: Path) -> None:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected = [col for col in numeric_columns if col != schema.target_column][:6]
    if not selected:
        return
    melted = df[selected].melt(var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=melted, x="feature", y="value", ax=ax)
    ax.set_title("Detection visuelle d'anomalies via boxplots")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
