"""End-to-end prediction for one trip + server-side scoring.

Hand-off point to Person 2's API:

    from pathsnap.inference import RoutePredictor, score_against_reference
    predictor = RoutePredictor("models/<model>.joblib", "models/decoder_config.json")
    out, trip = predictor.predict(trip_dir)                       # never reads the .route file
    out["evaluation"] = score_against_reference(trip_dir, out)     # server-side only

Pipeline: Person 1 loader -> Person 1 candidates + FEATURES -> classifier predict_proba
          -> HMM/Viterbi (hmm.py) -> shortest-path gap filling -> JSON (INTERFACE.md).
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np

from .hmm import HMMConfig, NetworkDistance, fill_route, viterbi_decode
from .metrics import route_continuity, route_overlap
from .team import FEATURES, candidates, load_for_prediction, load_with_route, point_targets


class RoutePredictor:
    def __init__(self, model_path, decoder_config_path):
        self.model = joblib.load(model_path)
        cfg = json.load(open(decoder_config_path))
        self.cfg = HMMConfig(**cfg["hmm"])
        self.model_name = cfg.get("model_name", "model")

    def predict(self, trip_dir, include_network=True):
        t0 = time.time()
        p1, trip = load_for_prediction(trip_dir)                # ground truth is NOT loaded
        cand = candidates(p1, align_route=False)
        t1 = time.time()
        prob = self.model.predict_proba(cand[FEATURES])[:, 1]
        t2 = time.time()
        c = cand.assign(p=prob)
        best = c.loc[c.groupby("point")["p"].idxmax()]
        clf = np.full(trip.n_points, -1, dtype=np.int64)
        clf[best["point"].to_numpy()] = best["arc"].to_numpy()
        net = NetworkDistance(trip, self.cfg.sp_limit)
        hmm, breaks = viterbi_decode(trip, cand, prob, self.cfg, net)
        route = fill_route(hmm, net, trip.arcs)
        t3 = time.time()

        def arc_coords(a):
            u, v = trip.arcs[a]
            return [[float(trip.nodes[u, 1]), float(trip.nodes[u, 0])],
                    [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]]

        out = {
            "trip_id": trip.trip_id,
            "model": self.model_name,
            "n_points": int(trip.n_points),
            "gps": [[float(la), float(lo)] for lo, la in trip.track[:, :2]],      # [lat, lon]
            "predicted_arcs_per_point": hmm.tolist(),
            "classifier_only_arcs_per_point": clf.tolist(),
            "predicted_route_arcs": [int(a) for a in route],
            "predicted_route_coords": _route_polyline(route, trip),               # [lat, lon]
            "decoder": {"chain_breaks": int(breaks),
                        "route_continuity_hmm": route_continuity(hmm, trip.arcs),
                        "route_continuity_classifier_only": route_continuity(clf, trip.arcs),
                        "gps_gaps_over_5s": int((np.diff(trip.t) > 5).sum())},
            "timing_ms": {"features": 1000 * (t1 - t0), "classifier": 1000 * (t2 - t1),
                          "viterbi": 1000 * (t3 - t2), "total": 1000 * (t3 - t0)},
        }
        if include_network:
            # road segments near the trajectory, for display only (input data, not ground truth)
            out["network_segments"] = [arc_coords(a) for a in np.unique(cand["arc"].to_numpy())]
        return out, trip


def _route_polyline(route, trip):
    if not route:
        return []
    pts = [trip.arcs[route[0], 0]] + [trip.arcs[a, 1] for a in route]
    return [[float(trip.nodes[n, 1]), float(trip.nodes[n, 0])] for n in pts]


def score_against_reference(trip_dir, out, partition="evaluator_test"):
    """SERVER-SIDE ONLY. Reads the .route file, aligns it with Person 1's alignment, and
    returns summary numbers only (no reference geometry or labels leave this function)."""
    p1, trip = load_with_route(trip_dir, partition)
    lab = point_targets(candidates(p1, align_route=True), trip.n_points)
    m = lab["reliable"]
    hmm = np.asarray(out["predicted_arcs_per_point"])
    clf = np.asarray(out["classifier_only_arcs_per_point"])
    res = {
        "points_scored": int(m.sum()),
        "accuracy_hmm": float((hmm[m] == lab["label_arc"][m]).mean()),
        "accuracy_classifier_only": float((clf[m] == lab["label_arc"][m]).mean()),
    }
    ov = route_overlap(np.asarray(out["predicted_route_arcs"]), trip.route, trip.arc_len)
    res.update({"route_precision": ov["route_precision"], "route_recall": ov["route_recall"],
                "route_f1": ov["route_f1"], "route_length_f1": ov["route_len_f1"]})
    return res, lab
