"""Per-trip candidate recall diagnostics for permitted labeled partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

from .features import candidate_feature_table
from .geometry import build_road_index
from .io import discover_trip_dirs, load_trip


def measure_coverage(data_root: Path, partitions: tuple[str, ...]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for partition in partitions:
        for trip_dir in discover_trip_dirs(data_root, partition):
            trip = load_trip(trip_dir, partition, include_route=True)
            if trip.route is None or trip.route.size == 0 or trip.track.size == 0:
                continue
            index = build_road_index(trip)
            candidates = candidate_feature_table(trip, index, align_route=True)
            if candidates.empty:
                continue
            obs = candidates.groupby(["trip_id", "obs_index"], sort=False).agg(
                target_arc_id=("target_arc_id", "first"),
                candidate_miss=("candidate_miss", "first"),
                nearest_fallback=("nearest_fallback", "first"),
                nearest_candidate_distance_m=("cross_track_m", "min"),
                candidate_count=("arc_id", "count"),
            )
            valid = obs[obs["target_arc_id"] >= 0]
            missed = valid[valid["candidate_miss"]]
            target_distances: list[float] = []
            for obs_index, row in missed.iterrows():
                _, point_index = obs_index
                point = Point(*index.projected_gps[int(point_index)])
                target_distances.append(
                    float(index.geometries[int(row["target_arc_id"])].distance(point))
                )
            positive_gaps = np.diff(trip.track[:, 2])
            positive_gaps = positive_gaps[positive_gaps > 0]
            records.append(
                {
                    "partition": partition,
                    "trip_id": trip.trip_id,
                    "observations": int(len(valid)),
                    "candidate_miss_observations": int(len(missed)),
                    "candidate_miss_rate": float(valid["candidate_miss"].mean()) if len(valid) else np.nan,
                    "nearest_fallback_observations": int(valid["nearest_fallback"].sum()),
                    "nearest_fallback_rate": float(valid["nearest_fallback"].mean()) if len(valid) else np.nan,
                    "candidate_count_median": float(valid["candidate_count"].median()) if len(valid) else np.nan,
                    "nearest_candidate_distance_p95_m": float(valid["nearest_candidate_distance_m"].quantile(0.95)) if len(valid) else np.nan,
                    "missed_target_distance_median": float(np.median(target_distances)) if target_distances else 0.0,
                    "missed_target_distance_p95_m": float(np.quantile(target_distances, 0.95)) if target_distances else 0.0,
                    "max_positive_timestamp_gap_s": float(positive_gaps.max()) if positive_gaps.size else np.nan,
                    "reference_route_arcs": int(len(trip.route)),
                }
            )
            print(f"Coverage {partition}/{trip.trip_id}: {len(missed)}/{len(valid)} candidate misses")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose reference-route candidate coverage by trip.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/eda/candidate_coverage_by_trip.csv"))
    args = parser.parse_args()
    coverage = measure_coverage(args.data_root, ("train", "student_test"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    for partition, group in coverage.groupby("partition", sort=True):
        obs = int(group["observations"].sum())
        misses = int(group["candidate_miss_observations"].sum())
        print(f"{partition}: {misses}/{obs} misses ({misses / obs:.3%})")


if __name__ == "__main__":
    main()
