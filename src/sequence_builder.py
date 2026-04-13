from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class SequenceDataset:
    X_seq: np.ndarray
    X_flat: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    groups: np.ndarray


def build_sequences(
    frame: pd.DataFrame,
    features: np.ndarray,
    target: np.ndarray,
    date_column: Optional[str],
    group_column: Optional[str],
    sequence_length: int,
) -> SequenceDataset:
    if group_column and group_column in frame.columns:
        group_series = frame[group_column].astype(str)
    else:
        group_series = pd.Series(["global"] * len(frame), index=frame.index)

    if date_column and date_column in frame.columns:
        date_series = frame[date_column]
    else:
        date_series = pd.Series(np.arange(len(frame)), index=frame.index)

    working_df = frame.copy()
    working_df["_group_key"] = group_series.values
    working_df["_date_key"] = date_series.values
    working_df["_row_position"] = np.arange(len(frame))
    working_df = working_df.sort_values(["_group_key", "_date_key"]).reset_index(drop=True)

    sequence_store: List[np.ndarray] = []
    flat_store: List[np.ndarray] = []
    target_store: List[float] = []
    date_store: List = []
    group_store: List[str] = []

    for group_name, group_df in working_df.groupby("_group_key", sort=False):
        idx = group_df["_row_position"].to_numpy()
        group_features = features[idx]
        group_target = target[idx]
        group_dates = group_df["_date_key"].to_numpy()

        if len(group_df) <= sequence_length:
            continue

        for end_idx in range(sequence_length, len(group_df)):
            start_idx = end_idx - sequence_length
            sequence_window = group_features[start_idx:end_idx]
            sequence_store.append(sequence_window)
            flat_store.append(sequence_window.reshape(-1))
            target_store.append(group_target[end_idx])
            date_store.append(group_dates[end_idx])
            group_store.append(group_name)

    if not sequence_store:
        raise ValueError("Aucune sequence n'a pu etre creee. Reduisez sequence_length.")

    return SequenceDataset(
        X_seq=np.asarray(sequence_store, dtype=np.float32),
        X_flat=np.asarray(flat_store, dtype=np.float32),
        y=np.asarray(target_store, dtype=np.float32),
        dates=np.asarray(date_store),
        groups=np.asarray(group_store),
    )
