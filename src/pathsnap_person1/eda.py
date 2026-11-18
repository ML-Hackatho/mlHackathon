"""Structured inventory and per-trip exploratory data summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .features import candidate_feature_table
from .geometry import build_road_index
from .io import discover_trip_dirs, load_trip
from .visualization import save_trip_map


def _file_summary(path: Path, columns: int, *, integral: bool = False) -> dict[str, int | bool]:
    if not path.is_file():
        return {"present": False, "bytes": 0, "rows": 0, "invalid_rows": 0}
    rows = 0
    invalid = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            rows += 1
            try:
                if len(parts) != columns:
                    raise ValueError
                values = [float(value) for value in parts]
                if not np.isfinite(values).all():
                    raise ValueError
                if integral and any(value != int(value) for value in values):
                    raise ValueError
            except (ValueError, OverflowError):
                invalid += 1
    return {"present": True, "bytes": path.stat().st_size, "rows": rows, "invalid_rows": invalid}


def _network_summary(n_nodes: int, arcs: np.ndarray) -> tuple[int, int]:
    if n_nodes == 0:
        return 0, 0
    if arcs.size == 0:
        return n_nodes, n_nodes
    sources = np.concatenate((arcs[:, 0], arcs[:, 1]))
    targets = np.concatenate((arcs[:, 1], arcs[:, 0]))
    adjacency = coo_matrix(
        (np.ones(len(sources), dtype=np.uint8), (sources, targets)),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    component_count = int(connected_components(adjacency, directed=False, return_labels=False))
    isolated_count = int(np.count_nonzero(np.diff(adjacency.indptr) == 0))
    return component_count, isolated_count


def summarize_trip(trip_dir: Path, partition: str) -> dict[str, object]:
    trip_id = trip_dir.name
    paths = {
        "track": trip_dir / f"{trip_id}.track",
        "nodes": trip_dir / f"{trip_id}.nodes",
        "arcs": trip_dir / f"{trip_id}.arcs",
        "route": trip_dir / f"{trip_id}.route",
    }
    file_stats = {
        "track": _file_summary(paths["track"], 3),
        "nodes": _file_summary(paths["nodes"], 2),
        "arcs": _file_summary(paths["arcs"], 2, integral=True),
        "route": _file_summary(paths["route"], 1, integral=True),
    }
    row: dict[str, object] = {
        "partition": partition,
        "trip_id": trip_id,
        "track_file_present": file_stats["track"]["present"],
        "nodes_file_present": file_stats["nodes"]["present"],
        "arcs_file_present": file_stats["arcs"]["present"],
        "route_file_present": file_stats["route"]["present"],
        "track_file_bytes": file_stats["track"]["bytes"],
        "nodes_file_bytes": file_stats["nodes"]["bytes"],
        "arcs_file_bytes": file_stats["arcs"]["bytes"],
        "route_file_bytes": file_stats["route"]["bytes"],
        "track_raw_rows": file_stats["track"]["rows"],
        "track_invalid_rows": file_stats["track"]["invalid_rows"],
        "nodes_raw_rows": file_stats["nodes"]["rows"],
        "nodes_invalid_rows": file_stats["nodes"]["invalid_rows"],
        "arcs_raw_rows": file_stats["arcs"]["rows"],
        "arcs_invalid_rows": file_stats["arcs"]["invalid_rows"],
        "route_raw_rows": file_stats["route"]["rows"],
        "route_invalid_rows": file_stats["route"]["invalid_rows"],
        "gps_observations": np.nan,
        "road_nodes": np.nan,
        "road_arcs": np.nan,
        "reference_route_arcs": np.nan,
        "track_invalid_or_impossible": np.nan,
        "longitude_min": np.nan,
        "longitude_max": np.nan,
        "latitude_min": np.nan,
        "latitude_max": np.nan,
        "timestamp_nonincreasing_steps": np.nan,
        "timestamp_non_1s_steps": np.nan,
        "time_gap_median_s": np.nan,
        "time_gap_p95_s": np.nan,
        "time_gap_max_s": np.nan,
        "road_undirected_components": np.nan,
        "road_isolated_nodes": np.nan,
        "load_error": "",
    }
    try:
        trip = load_trip(trip_dir, partition, include_route=True)
        row.update(
            {
                "gps_observations": int(len(trip.track)),
                "road_nodes": int(len(trip.nodes)),
                "road_arcs": int(len(trip.arcs)),
                "reference_route_arcs": int(len(trip.route)) if trip.route is not None else 0,
                "track_invalid_or_impossible": int(trip.quality["track_invalid_rows"]),
                "longitude_min": float(trip.track[:, 0].min()) if len(trip.track) else np.nan,
                "longitude_max": float(trip.track[:, 0].max()) if len(trip.track) else np.nan,
                "latitude_min": float(trip.track[:, 1].min()) if len(trip.track) else np.nan,
                "latitude_max": float(trip.track[:, 1].max()) if len(trip.track) else np.nan,
            }
        )
        dt = np.diff(trip.track[:, 2])
        positive_dt = dt[dt > 0]
        row.update(
            {
                "timestamp_nonincreasing_steps": int(np.count_nonzero(dt <= 0)),
                "timestamp_non_1s_steps": int(np.count_nonzero(np.abs(dt - 1.0) > 1e-6)),
                "time_gap_median_s": float(np.median(positive_dt)) if positive_dt.size else np.nan,
                "time_gap_p95_s": float(np.quantile(positive_dt, 0.95)) if positive_dt.size else np.nan,
                "time_gap_max_s": float(np.max(positive_dt)) if positive_dt.size else np.nan,
            }
        )
        components, isolated = _network_summary(len(trip.nodes), trip.arcs)
        row["road_undirected_components"] = components
        row["road_isolated_nodes"] = isolated
        row["load_error"] = ""
    except Exception as exc:  # Keep inventory rows even if one trip is malformed.
        row["load_error"] = f"{type(exc).__name__}: {exc}"
    return row


def build_inventory(data_root: Path, partitions: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for partition in partitions:
        try:
            trip_dirs = discover_trip_dirs(data_root, partition)
        except FileNotFoundError:
            continue
        rows.extend(summarize_trip(path, partition) for path in trip_dirs)
    return pd.DataFrame(rows)


def summarize_inventory(inventory: pd.DataFrame) -> dict[str, object]:
    partitions: dict[str, object] = {}
    for partition, part in inventory.groupby("partition", sort=True):
        valid = part[part["load_error"].isna() | part["load_error"].eq("")]
        total_steps = int((valid["gps_observations"].fillna(0).clip(lower=1) - 1).sum())
        irregular_steps = int(valid["timestamp_non_1s_steps"].fillna(0).sum())
        partitions[str(partition)] = {
            "trips": int(len(part)),
            "load_errors": int((part["load_error"].notna() & part["load_error"].ne("")).sum()),
            "gps_observations_total": int(valid["gps_observations"].fillna(0).sum()),
            "gps_observations_per_trip_median": float(valid["gps_observations"].median()),
            "gps_observations_per_trip_min": int(valid["gps_observations"].min()),
            "gps_observations_per_trip_max": int(valid["gps_observations"].max()),
            "invalid_or_impossible_gps_rows": int(valid["track_invalid_or_impossible"].fillna(0).sum()),
            "timestamp_nonincreasing_steps": int(valid["timestamp_nonincreasing_steps"].fillna(0).sum()),
            "timestamp_non_1s_steps": irregular_steps,
            "timestamp_steps_total": total_steps,
            "timestamp_exactly_1s_percent": (
                round(100.0 * (total_steps - irregular_steps) / total_steps, 3)
                if total_steps
                else None
            ),
            "road_nodes_total": int(valid["road_nodes"].fillna(0).sum()),
            "road_arcs_total": int(valid["road_arcs"].fillna(0).sum()),
            "road_components_per_trip_median": float(valid["road_undirected_components"].median()),
            "road_components_per_trip_max": int(valid["road_undirected_components"].max()),
            "isolated_road_nodes_total": int(valid["road_isolated_nodes"].fillna(0).sum()),
            "trips_with_reference_route": int(valid["route_file_present"].sum()),
            "longitude_extent": [
                float(valid["longitude_min"].min()),
                float(valid["longitude_max"].max()),
            ],
            "latitude_extent": [
                float(valid["latitude_min"].min()),
                float(valid["latitude_max"].max()),
            ],
            "trip_level_max_gap_median_s": float(valid["time_gap_max_s"].median()),
            "trip_level_max_gap_max_s": float(valid["time_gap_max_s"].max()),
        }
    return {
        "partitions": partitions,
        "inventory_csv": "trip_inventory.csv",
        "method_notes": [
            "Coordinates are read as longitude, latitude and projected per trip to metric AEQD coordinates for distance calculations.",
            "Road connectivity components are counted on an undirected view of directed arcs.",
            "GPS timestamps are analyzed in input order; non-increasing steps are reported separately.",
            "The route file is an ordered arc sequence and is not assumed to align row-for-row with GPS observations.",
        ],
    }


def save_eda_plot(inventory: pd.DataFrame, output_path: Path) -> None:
    """Write a compact actual-data summary plot for the report and team review."""
    usable = inventory[
        inventory["load_error"].isna() | inventory["load_error"].eq("")
    ].copy()
    if usable.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for partition, group in usable.groupby("partition", sort=True):
        axes[0].hist(
            group["gps_observations"].dropna(),
            alpha=0.65,
            bins=16,
            label=str(partition),
        )
    axes[0].set_title("GPS observations per trip")
    axes[0].set_xlabel("Observation count")
    axes[0].set_ylabel("Trips")
    axes[0].legend()

    for partition, group in usable.groupby("partition", sort=True):
        axes[1].scatter(
            group["road_arcs"],
            group["road_undirected_components"],
            alpha=0.7,
            label=str(partition),
        )
    axes[1].set_title("Road network size and components")
    axes[1].set_xlabel("Road arcs")
    axes[1].set_ylabel("Undirected connected components")
    axes[1].legend()

    axes[2].boxplot(
        [group["time_gap_max_s"].dropna().to_numpy() for _, group in usable.groupby("partition", sort=True)],
        tick_labels=[str(partition) for partition in sorted(usable["partition"].dropna().unique())],
        showfliers=True,
    )
    axes[2].set_title("Largest GPS gap per trip")
    axes[2].set_ylabel("Seconds")
    axes[2].set_yscale("log")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PathSnap trip inventory and EDA outputs.")
    parser.add_argument("--data-root", required=True, type=Path, help="Extracted folder containing train/ and student_test/.")
    parser.add_argument("--out", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--map-trip", help="Optional trip folder name for the representative Folium map.")
    parser.add_argument("--reuse-inventory", action="store_true", help="Reuse trip_inventory.csv already in --out instead of rereading every trip.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    inventory_path = args.out / "trip_inventory.csv"
    if args.reuse_inventory and inventory_path.is_file():
        inventory = pd.read_csv(
            inventory_path,
            dtype={"trip_id": str, "partition": str, "load_error": "string"},
        )
    else:
        inventory = build_inventory(args.data_root, ("train", "student_test"))
    if inventory.empty:
        raise FileNotFoundError("No trip folders found under train/ or student_test/ in --data-root")
    inventory.to_csv(inventory_path, index=False)
    save_eda_plot(inventory, args.out / "eda_summary.png")
    summary = summarize_inventory(inventory)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    map_partition = "train"
    candidates = inventory[
        (inventory["partition"] == map_partition)
        & (inventory["load_error"].isna() | inventory["load_error"].eq(""))
        & (inventory["gps_observations"].fillna(0) > 0)
        & (inventory["road_arcs"].fillna(0) > 0)
    ]
    if args.map_trip:
        candidates = inventory[
            (inventory["trip_id"] == args.map_trip)
            & (inventory["load_error"].isna() | inventory["load_error"].eq(""))
            & (inventory["gps_observations"].fillna(0) > 0)
            & (inventory["road_arcs"].fillna(0) > 0)
        ]
    if not candidates.empty:
        chosen = candidates.iloc[0]
        trip_dir = args.data_root / str(chosen["partition"]) / str(chosen["trip_id"])
        trip = load_trip(trip_dir, str(chosen["partition"]), include_route=True)
        index = build_road_index(trip)
        features = candidate_feature_table(trip, index, align_route=False)
        save_trip_map(trip, args.out / f"map_{trip.trip_id}.html", features)

    print(f"Wrote {args.out / 'trip_inventory.csv'}")
    print(f"Wrote {args.out / 'summary.json'}")
    print(f"Wrote {args.out / 'eda_summary.png'}")


if __name__ == "__main__":
    main()
