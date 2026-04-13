from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import DatasetSchema


@dataclass
class PreprocessingArtifacts:
    preprocessor: ColumnTransformer
    feature_names: List[str]
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def clean_air_quality_data(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    working_df = df.copy()
    working_df = working_df.drop_duplicates().reset_index(drop=True)

    if schema.date_column:
        working_df[schema.date_column] = pd.to_datetime(working_df[schema.date_column], errors="coerce")
        working_df = working_df.dropna(subset=[schema.date_column]).copy()

    working_df = working_df.dropna(subset=[schema.target_column]).copy()

    if schema.date_column and schema.group_column:
        working_df = working_df.sort_values([schema.group_column, schema.date_column]).reset_index(drop=True)
    elif schema.date_column:
        working_df = working_df.sort_values(schema.date_column).reset_index(drop=True)

    return add_time_features(working_df, schema.date_column)


def add_time_features(df: pd.DataFrame, date_column: Optional[str]) -> pd.DataFrame:
    if not date_column or date_column not in df.columns:
        return df

    enriched = df.copy()
    enriched["year"] = enriched[date_column].dt.year
    enriched["month"] = enriched[date_column].dt.month
    enriched["day"] = enriched[date_column].dt.day
    enriched["day_of_week"] = enriched[date_column].dt.dayofweek
    enriched["day_of_year"] = enriched[date_column].dt.dayofyear
    enriched["week_of_year"] = enriched[date_column].dt.isocalendar().week.astype(int)
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)

    if not enriched[date_column].dt.hour.isna().all():
        enriched["hour"] = enriched[date_column].dt.hour

    return enriched


def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_column: Optional[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Les ratios de split doivent sommer a 1.0.")

    if not date_column or date_column not in df.columns:
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

    unique_dates = np.array(sorted(df[date_column].dropna().unique()))
    n_dates = len(unique_dates)
    train_end = max(1, int(n_dates * train_ratio))
    val_end = max(train_end + 1, int(n_dates * (train_ratio + val_ratio)))

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    train_df = df[df[date_column].isin(train_dates)].copy()
    val_df = df[df[date_column].isin(val_dates)].copy()
    test_df = df[df[date_column].isin(test_dates)].copy()
    return train_df, val_df, test_df


def build_preprocessor(
    train_df: pd.DataFrame,
    schema: DatasetSchema,
) -> Tuple[ColumnTransformer, List[str]]:
    numeric_columns = [
        col for col in train_df.select_dtypes(include=["number"]).columns.tolist() if col != schema.target_column
    ]
    blocked_columns = {schema.target_column, schema.date_column}
    if schema.classification_target:
        blocked_columns.add(schema.classification_target)

    categorical_columns = [
        col
        for col in train_df.select_dtypes(include=["object", "category"]).columns.tolist()
        if col not in blocked_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    preprocessor.fit(train_df)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return preprocessor, feature_names


def fit_transform_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: DatasetSchema,
) -> PreprocessingArtifacts:
    preprocessor, feature_names = build_preprocessor(train_df, schema)

    X_train = np.asarray(preprocessor.transform(train_df), dtype=np.float32)
    X_val = np.asarray(preprocessor.transform(val_df), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(test_df), dtype=np.float32)

    y_train = train_df[schema.target_column].to_numpy(dtype=np.float32)
    y_val = val_df[schema.target_column].to_numpy(dtype=np.float32)
    y_test = test_df[schema.target_column].to_numpy(dtype=np.float32)

    return PreprocessingArtifacts(
        preprocessor=preprocessor,
        feature_names=feature_names,
        train_frame=train_df,
        val_frame=val_df,
        test_frame=test_df,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def save_preprocessor(preprocessor: ColumnTransformer, output_path: str) -> None:
    joblib.dump(preprocessor, output_path)
