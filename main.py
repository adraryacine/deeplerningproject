from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import infer_schema, load_dataset
from src.preprocessing import clean_air_quality_data
from src.train import TrainingConfig, run_training
from src.utils import FIGURES_DIR, ensure_directories
from src.visualization import generate_eda_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projet Deep Learning - Prediction de la qualite de l'air")
    parser.add_argument("--dataset", type=str, default=None, help="Chemin optionnel vers le dataset.")
    parser.add_argument("--sequence-length", type=int, default=14, help="Longueur des sequences temporelles.")
    parser.add_argument("--epochs", type=int, default=40, help="Nombre maximal d'epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille des batchs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Taux d'apprentissage Adam.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    raw_df = load_dataset(args.dataset)
    schema = infer_schema(raw_df, args.dataset)
    cleaned_df = clean_air_quality_data(raw_df, schema)
    generate_eda_plots(cleaned_df, schema, FIGURES_DIR)

    config = TrainingConfig(
        dataset_path=args.dataset,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    summary = run_training(config)

    print("Execution terminee.")
    print(f"Cible retenue : {summary['schema']['target_column']}")
    print(f"Type de probleme : {summary['schema']['task_type']}")
    print(f"Meilleur modele : {summary['best_model']}")
    print(f"Rapports disponibles dans : {Path('reports').resolve()}")


if __name__ == "__main__":
    main()
