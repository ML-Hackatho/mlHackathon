"""Geometry helpers.

Coordinates are WGS84 lon/lat (degrees). Each trip covers only a few km, so we
use a local equirectangular (tangent-plane) projection centred on the trip's road
network: x = R*dlon*cos(lat0), y = R*dlat. Over <20 km the distance error versus
the haversine great-circle distance is far below GPS noise (checked in EDA).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius (m)


@dataclass
class LocalProjection:
    lon0: float
    lat0: float

    @classmethod
    def from_lonlat(cls, lonlat: np.ndarray) -> "LocalProjection":
        return cls(float(np.mean(lonlat[:, 0])), float(np.mean(lonlat[:, 1])))

    def forward(self, lonlat: np.ndarray) -> np.ndarray:
        lon = np.radians(lonlat[:, 0] - self.lon0)
        lat = np.radians(lonlat[:, 1] - self.lat0)
        return np.column_stack([EARTH_R * lon * np.cos(np.radians(self.lat0)), EARTH_R * lat])

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        lon = np.degrees(xy[:, 0] / (EARTH_R * np.cos(np.radians(self.lat0)))) + self.lon0
        lat = np.degrees(xy[:, 1] / EARTH_R) + self.lat0
        return np.column_stack([lon, lat])


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Element-wise distance from points p to segments a->b (all (K,2)).
    Returns (distance, t_unclamped, t_clamped) where t is the projection parameter."""
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    L2 = np.where(L2 < 1e-9, 1e-9, L2)
    t = np.einsum("ij,ij->i", p - a, ab) / L2
    tc = np.clip(t, 0.0, 1.0)
    proj = a + tc[:, None] * ab
    d = np.hypot(p[:, 0] - proj[:, 0], p[:, 1] - proj[:, 1])
    return d, t, tc


def angle_diff(a, b):
    """Absolute smallest difference between angles (radians) in [0, pi]."""
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return d
