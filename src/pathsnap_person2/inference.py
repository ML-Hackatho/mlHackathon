"""Person 2 inference adapter: Person 1 candidates -> model -> Shreya's HMM."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from pathsnap_person1.features import FEATURES, candidate_feature_table
from pathsnap_person1.geometry import RoadIndex, build_road_index
from pathsnap_person1.io import TripData, load_trip

from .decoder import HMMConfig, NetworkDistance, collapse_sequence, fill_route, viterbi_decode


@dataclass
class DecoderTrip:
    """Minimal adapter exposing the geometry fields required by Shreya's HMM."""

    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    xy: np.ndarray
    arc_len: np.ndarray

    @property
    def n_points(self) -> int:
        return int(len(self.track))

    @property
    def t(self) -> np.ndarray:
        return self.track[:, 2]


def _continuity(arc_sequence: np.ndarray, arcs: np.ndarray) -> float:
    sequence = collapse_sequence(arc_sequence[arc_sequence >= 0])
    if len(sequence) < 2:
        return 1.0
    return float(np.mean(arcs[sequence[:-1], 1] == arcs[sequence[1:], 0]))


def _route_polyline(route_arcs: list[int], trip: TripData) -> list[list[float]]:
    if not route_arcs:
        return []
    nodes = [int(trip.arcs[route_arcs[0], 0])]
    nodes.extend(int(trip.arcs[arc_id, 1]) for arc_id in route_arcs)
    return [[float(trip.nodes[node, 1]), float(trip.nodes[node, 0])] for node in nodes]


def _arc_coords(arc_id: int, trip: TripData) -> list[list[float]]:
    start, end = trip.arcs[arc_id]
    return [
        [float(trip.nodes[start, 1]), float(trip.nodes[start, 0])],
        [float(trip.nodes[end, 1]), float(trip.nodes[end, 0])],
    ]


class RoutePredictor:
    """Load the selected classifier once and score a trip without reading .route."""

    def __init__(self, model_path: str | Path, decoder_config_path: str | Path):
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)
        config = json.loads(Path(decoder_config_path).read_text(encoding="utf-8"))
        self.cfg = HMMConfig(**config["hmm"])
        self.model_name = config.get("model_name", self.model_path.stem)

    def predict(
        self,
        trip_dir: str | Path,
        *,
        partition: str = "evaluator_test",
        include_network: bool = True,
    ) -> tuple[dict[str, object], TripData]:
        t0 = time.perf_counter()
        trip = load_trip(trip_dir, partition, include_route=False)
        road_index: RoadIndex = build_road_index(trip)
        candidates = candidate_feature_table(trip, road_index, align_route=False)
        t1 = time.perf_counter()
        if candidates.empty:
            raise ValueError(f"No road candidates generated for trip {trip.trip_id}")
        X = candidates[FEATURES].to_numpy(dtype=np.float32, copy=False)
        probability = self.model.predict_proba(X)[:, 1].astype(np.float64)
        t2 = time.perf_counter()

        scored = candidates[["point", "arc", "t_proj"]].copy()
        scored["probability"] = probability
        best_indices = scored.groupby("point", sort=False)["probability"].idxmax()
        best = scored.loc[best_indices]
        classifier_only = np.full(len(trip.track), -1, dtype=np.int64)
        classifier_only[best["point"].to_numpy(dtype=np.int64)] = best["arc"].to_numpy(dtype=np.int64)

        decoder_trip = DecoderTrip(
            trip_id=trip.trip_id,
            track=trip.track,
            nodes=trip.nodes,
            arcs=trip.arcs,
            xy=road_index.projected_gps,
            arc_len=road_index.lengths_m,
        )
        network = NetworkDistance(decoder_trip, self.cfg.sp_limit)
        predicted, chain_breaks = viterbi_decode(
            decoder_trip, scored[["point", "arc", "t_proj"]], probability, self.cfg, network
        )
        route_arcs = [int(a) for a in fill_route(predicted, network, trip.arcs)]
        t3 = time.perf_counter()

        out: dict[str, object] = {
            "trip_id": trip.trip_id,
            "model": self.model_name,
            "n_points": int(len(trip.track)),
            "gps": [[float(lat), float(lon)] for lon, lat in trip.track[:, :2]],
            "predicted_route_coords": _route_polyline(route_arcs, trip),
            "predicted_route_arcs": route_arcs,
            "predicted_arcs_per_point": predicted.astype(int).tolist(),
            "classifier_only_arcs_per_point": classifier_only.astype(int).tolist(),
            "decoder": {
                "chain_breaks": int(chain_breaks),
                "route_continuity_hmm": _continuity(predicted, trip.arcs),
                "route_continuity_classifier_only": _continuity(classifier_only, trip.arcs),
                "gps_gaps_over_5s": int((np.diff(trip.track[:, 2]) > 5).sum()),
            },
            "timing_ms": {
                "features": 1000.0 * (t1 - t0),
                "classifier": 1000.0 * (t2 - t1),
                "viterbi": 1000.0 * (t3 - t2),
                "total": 1000.0 * (t3 - t0),
            },
        }
        if include_network:
            used_arcs = np.unique(candidates["arc"].to_numpy(dtype=np.int64))
            out["network_segments"] = [_arc_coords(int(a), trip) for a in used_arcs]
        else:
            out["network_segments"] = []
        return out, trip


def _route_overlap(predicted: np.ndarray, reference: np.ndarray, arc_lengths: np.ndarray) -> dict[str, float]:
    pred_set = set(np.unique(predicted).astype(int).tolist())
    ref_set = set(np.unique(reference).astype(int).tolist())
    common = pred_set & ref_set

    def scores(weights: np.ndarray | None) -> tuple[float, float, float]:
        if weights is None:
            true_positive, predicted_total, reference_total = len(common), len(pred_set), len(ref_set)
        else:
            true_positive = float(weights[list(common)].sum()) if common else 0.0
            predicted_total = float(weights[list(pred_set)].sum()) if pred_set else 0.0
            reference_total = float(weights[list(ref_set)].sum()) if ref_set else 0.0
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / reference_total if reference_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    precision, recall, f1 = scores(None)
    _, _, length_f1 = scores(arc_lengths)
    return {"route_precision": precision, "route_recall": recall,
            "route_f1": f1, "route_length_f1": length_f1}


def score_against_reference(trip_dir: str | Path, out: dict[str, object], partition: str) -> dict[str, float | int] | None:
    """Server-side aggregate evaluation; prediction has already finished without .route."""
    trip = load_trip(trip_dir, partition, include_route=True)
    if trip.route is None or trip.route.size == 0:
        return None
    candidate_labels = candidate_feature_table(trip, align_route=True)
    if candidate_labels.empty or "target_arc_id" not in candidate_labels:
        return None
    targets = candidate_labels.drop_duplicates("point").set_index("point")["target_arc_id"]
    points = targets.index.to_numpy(dtype=np.int64)
    true_arc = targets.to_numpy(dtype=np.int64)
    valid = true_arc >= 0
    predicted = np.asarray(out["predicted_arcs_per_point"], dtype=np.int64)
    classifier_only = np.asarray(out["classifier_only_arcs_per_point"], dtype=np.int64)
    result: dict[str, float | int] = {
        "points_scored": int(valid.sum()),
        "accuracy_hmm": float(np.mean(predicted[points[valid]] == true_arc[valid])) if valid.any() else 0.0,
        "accuracy_classifier_only": float(np.mean(classifier_only[points[valid]] == true_arc[valid])) if valid.any() else 0.0,
    }
    result.update(_route_overlap(
        np.asarray(out["predicted_route_arcs"], dtype=np.int64), trip.route, build_road_index(trip).lengths_m
    ))
    return result
