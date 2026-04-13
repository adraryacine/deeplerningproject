from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .utils import discover_dataset_file


@dataclass
class DatasetSchema:
    dataset_path: str
    date_column: Optional[str]
    group_column: Optional[str]
    target_column: str
    task_type: str
    target_reason: str
    classification_target: Optional[str]
    numeric_features: List[str]
    categorical_features: List[str]
    pollutant_columns: List[str]


DATE_CANDIDATES = ["date", "datetime", "timestamp", "time", "day", "recorded_at"]
GROUP_CANDIDATES = ["city", "station", "location", "site"]
TARGET_PRIORITY = ["aqi", "air_quality_index"]
CLASSIFICATION_CANDIDATES = ["aqi_bucket", "aqi_category", "category", "label"]
POLLUTANT_HINTS = ["pm", "no", "nh3", "co", "so2", "o3", "benzene", "toluene", "xylene"]


def load_dataset(dataset_path: Optional[str] = None) -> pd.DataFrame:
    path = discover_dataset_file(dataset_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Format non supporte: {suffix}")

    return df


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def infer_schema(df: pd.DataFrame, dataset_path: Optional[str] = None) -> DatasetSchema:
    normalized_columns = {_normalize_name(col): col for col in df.columns}

    date_column = next((normalized_columns[c] for c in DATE_CANDIDATES if c in normalized_columns), None)
    group_column = next((normalized_columns[c] for c in GROUP_CANDIDATES if c in normalized_columns), None)

    target_column = None
    target_reason = ""
    for candidate in TARGET_PRIORITY:
        if candidate in normalized_columns:
            target_column = normalized_columns[candidate]
            target_reason = (
                f"La colonne '{target_column}' est numerique et correspond directement a l'indice de qualite de l'air. "
                "Elle est donc prioritaire comme cible principale."
            )
            break

    classification_target = next(
        (normalized_columns[c] for c in CLASSIFICATION_CANDIDATES if c in normalized_columns),
        None,
    )

    if target_column is None and classification_target is not None:
        target_column = classification_target
        target_reason = (
            f"Aucune colonne AQI numerique prioritaire n'a ete trouvee. "
            f"La cible retenue est donc '{classification_target}'."
        )

    if target_column is None:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_columns:
            raise ValueError("Impossible de detecter automatiquement une cible logique.")
        target_column = numeric_columns[-1]
        target_reason = (
            f"Aucune colonne AQI explicite n'a ete trouvee. La cible de secours est '{target_column}'."
        )

    task_type = "regression" if pd.api.types.is_numeric_dtype(df[target_column]) else "classification"

    numeric_features = [
        col for col in df.select_dtypes(include=["number"]).columns.tolist() if col != target_column
    ]
    leaked_targets = {target_column, date_column}
    if classification_target:
        leaked_targets.add(classification_target)

    categorical_features = [
        col
        for col in df.select_dtypes(include=["object", "category"]).columns.tolist()
        if col not in leaked_targets
    ]

    pollutant_columns = []
    for col in df.columns:
        lowered = _normalize_name(col)
        if any(hint in lowered for hint in POLLUTANT_HINTS):
            pollutant_columns.append(col)

    return DatasetSchema(
        dataset_path=str(Path(dataset_path).resolve()) if dataset_path else "",
        date_column=date_column,
        group_column=group_column,
        target_column=target_column,
        task_type=task_type,
        target_reason=target_reason,
        classification_target=classification_target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        pollutant_columns=pollutant_columns,
    )


def inspect_dataset(df: pd.DataFrame, schema: DatasetSchema) -> Dict:
    return {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().sort_values(ascending=False).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column": schema.target_column,
        "task_type": schema.task_type,
        "date_column": schema.date_column,
        "group_column": schema.group_column,
        "classification_target": schema.classification_target,
    }
