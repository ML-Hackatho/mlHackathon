"""Local metric geometry and spatial indexing for one trip's road network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .io import TripData


@dataclass
class RoadIndex:
    projected_gps: np.ndarray
    projected_nodes: np.ndarray
    geometries: list[LineString]
    tree: STRtree
    lengths_m: np.ndarray
    bearings_rad: np.ndarray
    from_degree: np.ndarray
    to_degree: np.ndarray
    transformer: Transformer


def project_trip(trip: TripData) -> tuple[np.ndarray, np.ndarray, Transformer]:
    """Project a local trip to azimuthal-equidistant metres around its midpoint.

    Each trip gets a local projection so distance and buffering are measured in
    metres, not angular degrees. This is intended for city-scale trajectories.
    """
    if trip.track.shape[0] == 0 or trip.nodes.shape[0] == 0:
        raise ValueError(f"Trip {trip.trip_id} has no usable GPS points or road nodes")
    lon0 = float(np.median(trip.track[:, 0]))
    lat0 = float(np.median(trip.track[:, 1]))
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.10f} +lon_0={lon0:.10f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(CRS.from_epsg(4326), local_crs, always_xy=True)
    gps_x, gps_y = transformer.transform(trip.track[:, 0], trip.track[:, 1])
    node_x, node_y = transformer.transform(trip.nodes[:, 0], trip.nodes[:, 1])
    return (
        np.column_stack((gps_x, gps_y)).astype(np.float64),
        np.column_stack((node_x, node_y)).astype(np.float64),
        transformer,
    )


def build_road_index(trip: TripData) -> RoadIndex:
    gps_xy, node_xy, transformer = project_trip(trip)
    if trip.arcs.shape[0] == 0:
        raise ValueError(f"Trip {trip.trip_id} has no road arcs")

    geometries: list[LineString] = []
    lengths: list[float] = []
    bearings: list[float] = []
    for from_node, to_node in trip.arcs:
        start = node_xy[int(from_node)]
        end = node_xy[int(to_node)]
        line = LineString((tuple(start), tuple(end)))
        geometries.append(line)
        dx, dy = end - start
        lengths.append(float(np.hypot(dx, dy)))
        bearings.append(float(np.arctan2(dy, dx)) if dx or dy else 0.0)

    from_degree = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes)).astype(float)
    to_degree = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes)).astype(float)
    return RoadIndex(
        projected_gps=gps_xy,
        projected_nodes=node_xy,
        geometries=geometries,
        tree=STRtree(geometries),
        lengths_m=np.asarray(lengths, dtype=float),
        bearings_rad=np.asarray(bearings, dtype=float),
        from_degree=from_degree[trip.arcs[:, 0]],
        to_degree=to_degree[trip.arcs[:, 1]],
        transformer=transformer,
    )


def nearby_arc_ids(
    index: RoadIndex,
    x: float,
    y: float,
    *,
    radius_m: float = 60.0,
    max_candidates: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return closest directed arcs within a metric radius, with nearest fallback."""
    point = Point(float(x), float(y))
    found = np.asarray(index.tree.query(point.buffer(radius_m)), dtype=np.int64).reshape(-1)
    if found.size == 0:
        # A candidate is still needed for every observation. This fallback is
        # recorded in the output so it can be diagnosed rather than hidden.
        nearest = index.tree.nearest(point)
        found = np.asarray([int(nearest)], dtype=np.int64)
    distances = np.asarray(
        [index.geometries[int(arc_id)].distance(point) for arc_id in found],
        dtype=np.float64,
    )
    order = np.lexsort((found, distances))[:max_candidates]
    return found[order], distances[order]


def point_on_arc(index: RoadIndex, arc_id: int, x: float, y: float) -> tuple[float, float]:
    point = Point(float(x), float(y))
    line = index.geometries[int(arc_id)]
    if line.length <= 1e-9:
        return 0.0, 0.0
    distance_along = float(line.project(point))
    return distance_along / float(line.length), distance_along
