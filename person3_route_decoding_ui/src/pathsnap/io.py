"""Road-network geometry container used by the HMM decoder.

All file reading goes through Person 1's loader (see team.py). This module only turns
the loaded arrays into the metre-based geometry the decoder needs (arc lengths,
bearings, projected GPS), keeping Person 1's observation order so that `point`
indices match the candidate table exactly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .geo import LocalProjection


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N,3) lon, lat, t   -- in Person 1's observation order
    nodes: np.ndarray            # (V,2) lon, lat
    arcs: np.ndarray             # (E,2) from, to (int)
    route: np.ndarray | None     # (R,) arc ids, only for development analysis
    proj: LocalProjection = field(repr=False, default=None)

    def __post_init__(self):
        self.arcs = np.asarray(self.arcs, dtype=np.int64)
        if self.proj is None:
            self.proj = LocalProjection.from_lonlat(self.nodes)
        self.xy = self.proj.forward(self.track[:, :2])
        self.node_xy = self.proj.forward(self.nodes)
        self.arc_a = self.node_xy[self.arcs[:, 0]]
        self.arc_b = self.node_xy[self.arcs[:, 1]]
        d = self.arc_b - self.arc_a
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        self.arc_bearing = np.arctan2(d[:, 1], d[:, 0])
        self.t = self.track[:, 2]

    @property
    def n_points(self):
        return len(self.track)


def list_trips(root: str) -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
