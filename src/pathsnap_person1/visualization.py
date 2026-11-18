"""Folium visualization for observed GPS, candidate road arcs, and reference route."""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd

from .io import TripData


def save_trip_map(
    trip: TripData,
    output_path: str | Path,
    candidate_table: pd.DataFrame | None = None,
    *,
    max_network_arcs: int = 5000,
) -> Path:
    if len(trip.track) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no valid GPS points to map")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    center = [float(trip.track[:, 1].mean()), float(trip.track[:, 0].mean())]
    map_view = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")

    gps_layer = folium.FeatureGroup(name="Observed GPS", show=True)
    gps_points = [[float(lat), float(lon)] for lon, lat in trip.track[:, :2]]
    folium.PolyLine(gps_points, color="#1769aa", weight=3, opacity=0.85, tooltip="GPS trace").add_to(gps_layer)
    stride = max(1, len(gps_points) // 200)
    for idx in range(0, len(gps_points), stride):
        folium.CircleMarker(
            gps_points[idx],
            radius=3,
            color="#1769aa",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"GPS observation {idx}",
        ).add_to(gps_layer)
    gps_layer.add_to(map_view)

    if candidate_table is not None and not candidate_table.empty:
        arc_ids = candidate_table["arc_id"].drop_duplicates().astype(int).tolist()[:max_network_arcs]
    else:
        arc_ids = list(range(min(len(trip.arcs), max_network_arcs)))
    network_layer = folium.FeatureGroup(name="Nearby road candidates", show=True)
    for arc_id in arc_ids:
        from_node, to_node = trip.arcs[arc_id]
        start_lon, start_lat = trip.nodes[int(from_node)]
        end_lon, end_lat = trip.nodes[int(to_node)]
        folium.PolyLine(
            [[float(start_lat), float(start_lon)], [float(end_lat), float(end_lon)]],
            color="#777777",
            weight=2,
            opacity=0.45,
            tooltip=f"Road arc {arc_id}",
        ).add_to(network_layer)
    network_layer.add_to(map_view)

    if trip.route is not None and trip.route.size:
        route_layer = folium.FeatureGroup(name="Reference route", show=True)
        route_parts: list[list[list[float]]] = []
        current_part: list[list[float]] = []
        previous_to_node: int | None = None
        for order, arc_id in enumerate(trip.route):
            from_node, to_node = trip.arcs[int(arc_id)]
            start_lon, start_lat = trip.nodes[int(from_node)]
            end_lon, end_lat = trip.nodes[int(to_node)]
            start = [float(start_lat), float(start_lon)]
            end = [float(end_lat), float(end_lon)]
            if current_part and previous_to_node != int(from_node):
                route_parts.append(current_part)
                current_part = []
            if not current_part:
                current_part.append(start)
            current_part.append(end)
            previous_to_node = int(to_node)
        if current_part:
            route_parts.append(current_part)
        for order, route_points in enumerate(route_parts):
            folium.PolyLine(
                route_points,
                color="#1b8f3a",
                weight=5,
                opacity=0.9,
                tooltip=f"Reference route part {order + 1} (ordered arc sequence)",
            ).add_to(route_layer)
        route_layer.add_to(map_view)

    folium.LayerControl(collapsed=False).add_to(map_view)
    south = min(point[0] for point in gps_points)
    north = max(point[0] for point in gps_points)
    west = min(point[1] for point in gps_points)
    east = max(point[1] for point in gps_points)
    if south != north and west != east:
        map_view.fit_bounds([[south, west], [north, east]], padding=(20, 20))
    map_view.save(str(output))
    return output
