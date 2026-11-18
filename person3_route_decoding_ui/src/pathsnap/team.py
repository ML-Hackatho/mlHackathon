"""Adapter to Person 1's data pipeline (src/pathsnap_person1 in the team repo).

Everything about reading trips, generating candidates, computing features and
aligning the reference route to GPS observations is Person 1's code. This module
only:
  * calls it with the agreed settings,
  * builds the decoder's geometry object from the same loaded arrays,
  * turns Person 1's per-candidate labels into per-point labels for scoring.
"""
from __future__ import annotations

import numpy as np

from . import config  # noqa: F401  (adds the team repo's src/ to sys.path)
from .io import Trip

from pathsnap_person1.features import FEATURES, candidate_feature_table  # noqa: E402
from pathsnap_person1.io import load_trip as person1_load_trip  # noqa: E402

__all__ = ["FEATURES", "load_for_prediction", "load_with_route", "candidates", "point_targets"]


def _to_trip(p1, with_route):
    return Trip(p1.trip_id, p1.track, p1.nodes, p1.arcs,
                p1.route if with_route else None)


def load_for_prediction(trip_dir, partition="evaluator_test"):
    """Prediction path: Person 1's loader with include_route=False (the .route file is never read)."""
    p1 = person1_load_trip(trip_dir, partition)
    return p1, _to_trip(p1, with_route=False)


def load_with_route(trip_dir, partition):
    """Development / server-side scoring only."""
    p1 = person1_load_trip(trip_dir, partition, include_route=True)
    return p1, _to_trip(p1, with_route=True)


def candidates(p1, align_route=False):
    """Person 1's candidate table (one row per point/arc, ordered by point)."""
    return candidate_feature_table(p1, align_route=align_route)


def point_targets(cand, n_points):
    """Per-point aligned reference arc from Person 1's labels (-1 if unavailable).
    `reliable` marks points that have a label; candidate-miss points stay in (they
    count as errors), following Person 1's evaluation rule."""
    label_arc = np.full(n_points, -1, dtype=np.int64)
    if "target_arc_id" in cand:
        first = cand.drop_duplicates("point")
        label_arc[first["point"].to_numpy()] = first["target_arc_id"].to_numpy()
    return {"label_arc": label_arc, "reliable": label_arc >= 0}
