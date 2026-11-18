"""Readers for the documented PathSnap per-trip text format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TripData:
    trip_id: str
    partition: str
    path: Path
    track: np.ndarray  # columns: longitude, latitude, timestamp_seconds
    track_source_index: np.ndarray  # zero-based nonblank input-record index
    nodes: np.ndarray  # columns: longitude, latitude; row index is node_id
    arcs: np.ndarray  # columns: from_node_id, to_node_id; row index is arc_id
    route: np.ndarray | None  # chronological arc_id sequence, if supplied
    quality: dict[str, int]


def _read_numeric_rows(
    path: Path,
    columns: int,
    *,
    integral: bool = False,
    strict: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Read whitespace-separated data and return values, invalid count, row count.

    Invalid track rows can be skipped because track rows do not define IDs. Invalid
    node or arc rows must raise: their row positions define node_id and arc_id.
    """
    values: list[list[float]] = []
    invalid = 0
    records = 0
    bad_examples: list[str] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            parts = line.split()
            if not parts:
                continue
            records += 1
            try:
                if len(parts) != columns:
                    raise ValueError(f"expected {columns} columns, found {len(parts)}")
                row = [float(value) for value in parts]
                if not np.isfinite(row).all():
                    raise ValueError("non-finite value")
                if integral and any(value != int(value) for value in row):
                    raise ValueError("expected integer values")
            except (TypeError, ValueError, OverflowError) as exc:
                invalid += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"line {line_number}: {exc}")
                continue
            values.append(row)

    if strict and invalid:
        detail = "; ".join(bad_examples)
        raise ValueError(f"{path}: {invalid} invalid row(s); {detail}")
    dtype = np.int64 if integral else np.float64
    array = np.asarray(values, dtype=dtype)
    if array.size == 0:
        array = np.empty((0, columns), dtype=dtype)
    return array.reshape((-1, columns)), invalid, records


def discover_trip_dirs(data_root: str | Path, partition: str) -> list[Path]:
    """Find trip folders in an extracted dataset root (for example Student folder)."""
    root = Path(data_root).expanduser().resolve()
    partition_root = root / partition
    if not partition_root.is_dir():
        raise FileNotFoundError(
            f"Partition folder not found: {partition_root}. Set --data-root to the "
            "extracted folder containing train/ and student_test/."
        )
    return sorted(
        (path for path in partition_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )


def load_trip(
    trip_dir: str | Path,
    partition: str,
    *,
    include_route: bool = False,
) -> TripData:
    """Load a trip, excluding reference labels unless explicitly requested."""
    folder = Path(trip_dir).expanduser().resolve()
    trip_id = folder.name
    track_path = folder / f"{trip_id}.track"
    nodes_path = folder / f"{trip_id}.nodes"
    arcs_path = folder / f"{trip_id}.arcs"
    route_path = folder / f"{trip_id}.route"
    for required in (track_path, nodes_path, arcs_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required trip file missing: {required}")

    track, bad_track, raw_track = _read_numeric_rows(track_path, 3, strict=False)
    nodes, bad_nodes, raw_nodes = _read_numeric_rows(nodes_path, 2, strict=True)
    arcs, bad_arcs, raw_arcs = _read_numeric_rows(
        arcs_path, 2, integral=True, strict=True
    )
    if include_route and route_path.is_file():
        route_rows, bad_route, raw_route = _read_numeric_rows(
            route_path, 1, integral=True, strict=True
        )
        route = route_rows[:, 0].astype(np.int64)
    else:
        bad_route, raw_route = 0, 0
        route = None

    # GPS coordinate order is lon, lat in the supplied format. Drop impossible
    # coordinates, preserving the mapping to their original source records.
    track_source = np.arange(raw_track - bad_track, dtype=np.int64)
    if track.shape[0] != track_source.shape[0]:
        # Reconstruct valid record indices when malformed lines were skipped.
        valid_records: list[int] = []
        record_index = 0
        with track_path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.split():
                    continue
                parts = line.split()
                try:
                    row = [float(value) for value in parts]
                    valid = len(row) == 3 and np.isfinite(row).all()
                except (ValueError, OverflowError):
                    valid = False
                if valid:
                    valid_records.append(record_index)
                record_index += 1
        track_source = np.asarray(valid_records, dtype=np.int64)
    geo_valid = (
        (track[:, 0] >= -180)
        & (track[:, 0] <= 180)
        & (track[:, 1] >= -90)
        & (track[:, 1] <= 90)
    )
    if not geo_valid.all():
        track = track[geo_valid]
        track_source = track_source[geo_valid]

    if not np.isfinite(nodes).all() or np.any(np.abs(nodes[:, 0]) > 180) or np.any(
        np.abs(nodes[:, 1]) > 90
    ):
        raise ValueError(f"{nodes_path}: invalid longitude/latitude; node IDs cannot be renumbered")
    if arcs.size and (np.any(arcs < 0) or np.any(arcs >= len(nodes))):
        raise ValueError(f"{arcs_path}: arc endpoint references a missing node")
    if route is not None and route.size and (np.any(route < 0) or np.any(route >= len(arcs))):
        raise ValueError(f"{route_path}: route references a missing arc")

    return TripData(
        trip_id=trip_id,
        partition=partition,
        path=folder,
        track=track,
        track_source_index=track_source,
        nodes=nodes,
        arcs=arcs,
        route=route,
        quality={
            "track_raw_rows": raw_track,
            "track_invalid_rows": bad_track + int((~geo_valid).sum()),
            "nodes_raw_rows": raw_nodes,
            "nodes_invalid_rows": bad_nodes,
            "arcs_raw_rows": raw_arcs,
            "arcs_invalid_rows": bad_arcs,
            "route_raw_rows": raw_route,
            "route_invalid_rows": bad_route,
        },
    )
