"""Compute classifier-only directed continuity for the report's model table.

This uses the 10 student-test candidate exports and only each trip's road arcs;
it does not load the reference .route files. The two Person 2 baseline models
are refit with the same settings recorded in their experiment outputs because
only their configurations (not fitted weights) were saved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pathsnap_person1.io import load_trip
from pathsnap_person1.features import FEATURES
from pathsnap_person2.inference import _continuity
from pathsnap_person2.training import _fit_full, _read_candidate_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--person3-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    person3 = args.person3_root.resolve()
    data_root = args.data_root.resolve()
    candidate_dir = root / "outputs" / "shivanand_handoff"
    train = _read_candidate_table(candidate_dir / "train_candidates.csv.gz")
    student = _read_candidate_table(candidate_dir / "student_test_candidates.csv.gz")
    train_hits = train.loc[~train["candidate_miss"]].reset_index(drop=True)

    model_paths = {
        "logistic_regression": root / "outputs/baselines/models/logistic_regression.joblib",
        "random_forest": root / "outputs/baselines/models/random_forest.joblib",
        "mlp": person3 / "models/mlp.joblib",
        "xgboost_tuned": root / "outputs/person2/models/final_model.joblib",
    }
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    models["lightgbm"] = _fit_full("lightgbm", {}, train_hits)[0]
    models["xgboost"] = _fit_full("xgboost", {}, train_hits)[0]

    X = student[FEATURES].to_numpy(dtype=np.float32, copy=False)

    trip_arcs = {}
    for trip_id in student["trip_id"].drop_duplicates().astype(str):
        trip = load_trip(data_root / "student_test" / trip_id, "student_test", include_route=False)
        trip_arcs[trip_id] = trip.arcs

    trip_rows: list[dict[str, object]] = []
    for name, model in models.items():
        probability = model.predict_proba(X)[:, 1]
        scored = student[["trip_id", "point", "arc"]].copy()
        scored["probability"] = probability
        for trip_id, group in scored.groupby("trip_id", sort=False):
            best = group.loc[group.groupby("point", sort=False)["probability"].idxmax()]
            best = best.sort_values("point", kind="stable")
            trip_rows.append({
                "model": name,
                "trip_id": str(trip_id),
                "observations": int(best.shape[0]),
                "classifier_only_route_continuity": _continuity(
                    best["arc"].to_numpy(dtype=np.int64), trip_arcs[str(trip_id)]
                ),
            })

    per_trip = pd.DataFrame(trip_rows)
    summary = (
        per_trip.groupby("model", as_index=False)
        .agg(
            student_test_route_continuity_mean_per_trip=("classifier_only_route_continuity", "mean"),
            student_test_route_continuity_std_per_trip=("classifier_only_route_continuity", "std"),
            student_test_trips=("trip_id", "nunique"),
        )
    )
    output = args.output or root / "reports" / "model_route_continuity_student_test.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    per_trip.to_csv(output.with_name("model_route_continuity_student_test_by_trip.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
