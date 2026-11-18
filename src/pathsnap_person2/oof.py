"""Create row-aligned out-of-fold probabilities for the tuned final model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .training import (
    _make_estimator,
    _positive_weight,
    _read_candidate_table,
    _trip_folds,
)

LOG = logging.getLogger("pathsnap_person2.oof")


def export_oof(candidate_dir: Path, output_dir: Path, folds_n: int = 5) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = _read_candidate_table(candidate_dir / "train_candidates.csv.gz")
    config_path = output_dir / "models" / "final_model.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Train/tune a final model first: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    name = config["model_name"]
    params = config["tuning"]["best_params"]
    X = table[[*config["feature_columns"]]].to_numpy(dtype=np.float32, copy=False)
    y = table["is_target"].astype(np.uint8).to_numpy()
    trip = table["trip_id"].astype(str).to_numpy()
    miss = table["candidate_miss"].to_numpy(dtype=bool)
    folds = _trip_folds(table, folds_n)
    probability = np.full(len(table), np.nan, dtype=np.float32)
    for fold_no, (fit_trips, valid_trips) in enumerate(folds, start=1):
        fit_idx = np.flatnonzero(np.isin(trip, fit_trips) & ~miss)
        valid_idx = np.flatnonzero(np.isin(trip, valid_trips))
        model = _make_estimator(name, params, _positive_weight(y[fit_idx]))
        model.fit(X[fit_idx], y[fit_idx])
        probability[valid_idx] = model.predict_proba(X[valid_idx])[:, 1].astype(np.float32)
        LOG.info("OOF fold %d/%d written for %d trip(s)", fold_no, len(folds), len(valid_trips))
    if not np.isfinite(probability).all():
        raise RuntimeError("OOF output has missing predictions; check candidate row grouping")
    output_path = output_dir / f"oof_{name}.npy"
    np.save(output_path, probability)
    metadata = {
        "file": output_path.name,
        "model_name": name,
        "candidate_csv": str((candidate_dir / "train_candidates.csv.gz").resolve()),
        "row_count": int(len(table)),
        "row_order": "exact row order of train_candidates.csv.gz after label_available filtering",
        "folds": int(len(folds)),
        "group_column": "trip_id",
        "candidate_miss_handling": "missed observation rows are predicted in held-out fold; excluded from fit",
    }
    (output_dir / f"oof_{name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOG.info("Saved %d OOF probabilities to %s", len(probability), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final-model OOF probabilities in candidate CSV row order.")
    parser.add_argument("--candidate-dir", type=Path, default=Path("outputs/shivanand_handoff"))
    parser.add_argument("--out", type=Path, default=Path("outputs/person2"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print(export_oof(args.candidate_dir, args.out, args.folds))


if __name__ == "__main__":
    main()
