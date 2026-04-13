import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_directories() -> None:
    for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def discover_dataset_file(preferred_name: Optional[str] = None) -> Path:
    search_roots = [PROJECT_ROOT, DATA_DIR]

    if preferred_name:
        candidate = Path(preferred_name)
        if candidate.is_file():
            return candidate.resolve()
        for root in search_roots:
            candidate = root / preferred_name
            if candidate.is_file():
                return candidate.resolve()

    patterns = ["*.csv", "*.xlsx", "*.xls", "*.parquet"]
    matches: List[Path] = []
    for root in search_roots:
        for pattern in patterns:
            matches.extend(root.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            "Aucun dataset trouve. Placez un fichier tabulaire dans la racine du projet ou dans data/."
        )

    matches = sorted(set(path.resolve() for path in matches))
    return matches[0]
