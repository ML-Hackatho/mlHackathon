"""Road candidate generation, route-label alignment, and numeric features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

from .geometry import RoadIndex, build_road_index, nearby_arc_ids, point_on_arc
from .io import TripData


FEATURE_COLUMNS = [
    "cross_track_m",
    "gps_step_m",
    "next_gps_step_m",
    "time_gap_s",
    "gps_speed_mps",
    "arc_length_m",
    "projection_fraction",
    "heading_cos",
    "heading_sin",
    "prev_heading_cos",
    "next_heading_cos",
    "prev_heading_available",
    "next_heading_available",
    "from_degree",
    "to_degree",
]


def _angle_difference(left: float, right: float) -> float:
    return (left - right + math.pi) % (2 * math.pi) - math.pi


def _direction(start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    dx, dy = end - start
    distance = float(np.hypot(dx, dy))
    return (float(np.arctan2(dy, dx)), distance) if distance > 1e-9 else (0.0, 0.0)


def align_route_to_observations(
    trip: TripData,
    index: RoadIndex | None = None,
    *,
    search_radius_m: float = 150.0,
    states_per_observation: int = 12,
) -> np.ndarray:
    """Associate each GPS point with an ordered reference-route arc.

    The route file is a sequence of traversed arc IDs, not one label per GPS
    row. This aligns it to observations using metric distance and a soft
    progression penalty. It deliberately does not use row-number matching.
    Review the resulting alignment with maps before treating it as unquestioned
    ground truth; supplied route labels may themselves have limitations.
    """
    if trip.route is None or trip.route.size == 0:
        return np.full(trip.track.shape[0], -1, dtype=np.int64)
    if index is None:
        index = build_road_index(trip)

    route_geometries = [index.geometries[int(arc_id)] for arc_id in trip.route]
    route_tree = STRtree(route_geometries)
    route_lengths = np.asarray([geom.length for geom in route_geometries], dtype=float)
    route_offsets = np.concatenate(([0.0], np.cumsum(route_lengths[:-1])))
    all_states: list[list[tuple[int, int, float, float]]] = []

    for x, y in index.projected_gps:
        point = Point(float(x), float(y))
        route_positions = np.asarray(
            route_tree.query(point.buffer(search_radius_m)), dtype=np.int64
        ).reshape(-1)
        if route_positions.size == 0:
            route_positions = np.asarray([int(route_tree.nearest(point))], dtype=np.int64)
        distances = np.asarray(
            [route_geometries[int(pos)].distance(point) for pos in route_positions],
            dtype=float,
        )
        order = np.lexsort((route_positions, distances))[:states_per_observation]
        states: list[tuple[int, int, float, float]] = []
        for raw_idx in order:
            route_pos = int(route_positions[raw_idx])
            geometry = route_geometries[route_pos]
            along = float(geometry.project(point)) if geometry.length > 0 else 0.0
            progress = float(route_offsets[route_pos] + along)
            states.append(
                (route_pos, int(trip.route[route_pos]), progress, float(distances[raw_idx]))
            )
        all_states.append(states)

    if not all_states:
        return np.empty(0, dtype=np.int64)

    # Viterbi-style alignment over nearby ordered route positions. The soft
    # backward penalty permits noisy observations while strongly preferring
    # chronological route progress.
    costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    sigma_m = 20.0
    for obs_idx, states in enumerate(all_states):
        emission = np.asarray([(state[3] / sigma_m) ** 2 for state in states], dtype=float)
        if obs_idx == 0:
            costs.append(emission)
            backpointers.append(np.full(len(states), -1, dtype=np.int64))
            continue

        previous_states = all_states[obs_idx - 1]
        previous_cost = costs[-1]
        gps_step = float(
            np.linalg.norm(index.projected_gps[obs_idx] - index.projected_gps[obs_idx - 1])
        )
        dt = float(trip.track[obs_idx, 2] - trip.track[obs_idx - 1, 2])
        dt = max(dt, 1.0)
        scale = max(12.0, gps_step * 0.5 + 8.0)
        current_cost = np.empty(len(states), dtype=float)
        current_back = np.zeros(len(states), dtype=np.int64)
        for cur_idx, current in enumerate(states):
            best_cost = float("inf")
            best_previous = 0
            for prev_idx, previous in enumerate(previous_states):
                route_delta = current[2] - previous[2]
                backward_metres = max(0.0, -route_delta)
                backward_positions = max(0, previous[0] - current[0])
                transition = 0.20 * abs(route_delta - gps_step) / scale
                transition += 2.0 * backward_metres / scale
                transition += 25.0 * backward_positions
                # Discourage a route jump that cannot plausibly fit in the time
                # gap, while allowing extra room for noisy or sparse GPS.
                plausible_limit = 55.0 * dt + 150.0
                transition += max(0.0, route_delta - plausible_limit) / 30.0
                total = float(previous_cost[prev_idx] + transition + emission[cur_idx])
                if total < best_cost:
                    best_cost = total
                    best_previous = prev_idx
            current_cost[cur_idx] = best_cost
            current_back[cur_idx] = best_previous
        costs.append(current_cost)
        backpointers.append(current_back)

    selected = np.zeros(len(all_states), dtype=np.int64)
    selected[-1] = int(np.argmin(costs[-1]))
    for obs_idx in range(len(all_states) - 1, 0, -1):
        selected[obs_idx - 1] = backpointers[obs_idx][selected[obs_idx]]
    return np.asarray(
        [all_states[i][selected[i]][1] for i in range(len(all_states))], dtype=np.int64
    )


def candidate_feature_table(
    trip: TripData,
    index: RoadIndex | None = None,
    *,
    radius_m: float = 60.0,
    max_candidates: int = 12,
    align_route: bool = False,
) -> pd.DataFrame:
    """Build candidate rows; label alignment is opt-in for development data only."""
    if index is None:
        index = build_road_index(trip)
    gps = index.projected_gps
    timestamps = trip.track[:, 2]
    if align_route and trip.route is not None:
        target_arc = align_route_to_observations(trip, index)
    else:
        target_arc = np.full(len(gps), -1, dtype=np.int64)

    rows: list[dict[str, float | int | str | bool]] = []
    for obs_idx, (x, y) in enumerate(gps):
        candidate_ids, candidate_distances = nearby_arc_ids(
            index,
            float(x),
            float(y),
            radius_m=radius_m,
            max_candidates=max_candidates,
        )
        prev_heading = 0.0
        next_heading = 0.0
        prev_distance = 0.0
        next_distance = 0.0
        prev_dt = 0.0
        next_dt = 0.0
        if obs_idx > 0:
            prev_heading, prev_distance = _direction(gps[obs_idx - 1], gps[obs_idx])
            prev_dt = max(float(timestamps[obs_idx] - timestamps[obs_idx - 1]), 0.0)
        if obs_idx + 1 < len(gps):
            next_heading, next_distance = _direction(gps[obs_idx], gps[obs_idx + 1])
            next_dt = max(float(timestamps[obs_idx + 1] - timestamps[obs_idx]), 0.0)
        current_target = int(target_arc[obs_idx])
        candidate_miss = current_target >= 0 and current_target not in set(candidate_ids.tolist())

        for arc_id, cross_track in zip(candidate_ids, candidate_distances):
            arc_id = int(arc_id)
            bearing = float(index.bearings_rad[arc_id])
            prev_delta = _angle_difference(bearing, prev_heading)
            next_delta = _angle_difference(bearing, next_heading)
            projection_fraction, _ = point_on_arc(index, arc_id, float(x), float(y))
            rows.append(
                {
                    "trip_id": trip.trip_id,
                    "partition": trip.partition,
                    "obs_index": int(obs_idx),
                    "source_track_row": int(trip.track_source_index[obs_idx]),
                    "arc_id": arc_id,
                    "target_arc_id": current_target,
                    "label_available": current_target >= 0,
                    "is_target": current_target >= 0 and arc_id == current_target,
                    "candidate_miss": bool(candidate_miss),
                    "nearest_fallback": bool(float(cross_track) > radius_m),
                    "cross_track_m": float(cross_track),
                    "gps_step_m": prev_distance,
                    "next_gps_step_m": next_distance,
                    "time_gap_s": prev_dt,
                    "gps_speed_mps": prev_distance / prev_dt if prev_dt > 0 else 0.0,
                    "arc_length_m": float(index.lengths_m[arc_id]),
                    "projection_fraction": projection_fraction,
                    "heading_cos": math.cos(prev_delta),
                    "heading_sin": math.sin(prev_delta),
                    "prev_heading_cos": math.cos(prev_delta),
                    "next_heading_cos": math.cos(next_delta),
                    "prev_heading_available": int(obs_idx > 0 and prev_dt > 0),
                    "next_heading_available": int(obs_idx + 1 < len(gps) and next_dt > 0),
                    "from_degree": float(index.from_degree[arc_id]),
                    "to_degree": float(index.to_degree[arc_id]),
                }
            )
    return pd.DataFrame(rows)
