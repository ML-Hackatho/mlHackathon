"""Export labeled candidate tables for the team's train and student-test partitions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from .features import FEATURE_COLUMNS, candidate_feature_table
from .io import discover_trip_dirs, load_trip


FEATURE_DEFINITIONS = {
    "cross_track_m": "Metric distance from the GPS observation to the directed candidate arc.",
    "gps_step_m": "Projected straight-line distance from the previous valid GPS observation to this observation; zero for the first observation.",
    "next_gps_step_m": "Projected straight-line distance from this observation to the next valid GPS observation; zero for the final observation.",
    "time_gap_s": "Positive timestamp difference from the previous valid observation; zero when unavailable or non-positive.",
    "gps_speed_mps": "gps_step_m divided by time_gap_s; zero when the previous step has no positive time gap.",
    "arc_length_m": "Projected straight-line length between the candidate arc's from-node and to-node.",
    "projection_fraction": "Closest-point fraction along the directed arc, clamped to [0, 1]; 0 is the from-node and 1 is the to-node.",
    "heading_cos": "Cosine of the wrapped angle from candidate-arc bearing to the incoming GPS heading.",
    "heading_sin": "Sine of the wrapped angle from candidate-arc bearing to the incoming GPS heading.",
    "prev_heading_cos": "Same value as heading_cos in the current implementation; retained for compatibility with Person 1 baseline models.",
    "next_heading_cos": "Cosine of the wrapped angle from candidate-arc bearing to the outgoing GPS heading.",
    "prev_heading_available": "1 when a previous observation and positive timestamp gap provide an incoming heading; otherwise 0.",
    "next_heading_available": "1 when a next observation and positive timestamp gap provide an outgoing heading; otherwise 0.",
    "from_degree": "Directed out-degree of the candidate arc's from-node.",
    "to_degree": "Directed in-degree of the candidate arc's to-node.",
}


def _export_partition(data_root: Path, partition: str, output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observation_count = 0
    candidate_count = 0
    candidate_miss_count = 0
    labeled_trip_count = 0
    first = True

    with gzip.open(output_path, mode="wt", compresslevel=3, encoding="utf-8", newline="") as stream:
        for trip_dir in discover_trip_dirs(data_root, partition):
            trip = load_trip(trip_dir, partition, include_route=True)
            if trip.route is None or trip.route.size == 0 or trip.track.size == 0:
                print(f"Skip {partition}/{trip.trip_id}: missing route or usable GPS observations")
                continue
            table = candidate_feature_table(trip, align_route=True)
            if table.empty:
                print(f"Skip {partition}/{trip.trip_id}: no road candidates")
                continue
            table.to_csv(stream, index=False, header=first, lineterminator="\n")
            first = False
            by_observation = table.groupby("obs_index", sort=False).agg(
                candidate_miss=("candidate_miss", "first"),
                label_available=("label_available", "first"),
            )
            observation_count += int(by_observation["label_available"].sum())
            candidate_miss_count += int(by_observation["candidate_miss"].sum())
            candidate_count += int(len(table))
            labeled_trip_count += 1
            print(
                f"Exported {partition}/{trip.trip_id}: "
                f"{len(by_observation)} observations, {len(table)} candidate rows"
            )

    return {
        "partition": partition,
        "trips_exported": labeled_trip_count,
        "observations": observation_count,
        "candidate_rows": candidate_count,
        "candidate_miss_observations": candidate_miss_count,
        "candidate_miss_rate": candidate_miss_count / observation_count if observation_count else None,
        "file": output_path.name,
        "compressed_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export actual labeled candidate feature tables.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/shivanand_handoff"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = [
        _export_partition(
            args.data_root,
            partition,
            args.out / f"{partition}_candidates.csv.gz",
        )
        for partition in ("train", "student_test")
    ]
    schema = {
        "row_grain": "one row per (trip_id, obs_index, arc_id) candidate",
        "order": "trips sorted by directory name; observations in input time order; candidates nearest-to-farthest within each observation",
        "interface_columns": ["point", "arc", "t_proj"],
        "internal_identity_columns": ["trip_id", "obs_index", "arc_id"],
        "model_input_columns": FEATURE_COLUMNS,
        "feature_definitions": FEATURE_DEFINITIONS,
        "target_column": "is_target",
        "target_format": "boolean; True iff this candidate arc equals the route-to-GPS aligned target arc",
        "label_columns": ["target_arc_id", "label_available", "is_target", "candidate_miss"],
        "candidate_miss_handling": "Keep missed observations in validation to report candidate miss rate; exclude all candidate rows from missed observations during model fitting.",
        "candidate_settings": {"radius_m": 60.0, "max_candidates": 12, "nearest_fallback": True},
        "alignment": "The ordered route arc sequence is spatially aligned to GPS observations with a soft route-progression penalty; it is not row-index matched.",
        "inference": {
            "load_route": False,
            "candidate_feature_table_align_route": False,
            "label_columns_present": False,
            "model_input": "candidate_table[model_input_columns]",
            "positive_probability": "predict_proba(X)[:, 1]",
            "grouping_and_arc_mapping": ["trip_id", "obs_index", "arc_id"],
        },
        "partition_use": {
            "train": "fit and group-aware validation only",
            "student_test": "development evaluation only; do not fit on it",
            "evaluator_test": "not exported or read by this command",
        },
        "files": summaries,
    }
    (args.out / "candidate_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.out / 'candidate_schema.json'}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
