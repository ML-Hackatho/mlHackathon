"""HMM map matching with Viterbi decoding.

Hidden state at time t : one candidate arc c for GPS point t (plus the projected
                         position s_c of the point on that arc, in metres).
Emission               : log P(z_t | c) = w_emit * log(p_clf(c | point t)), where p_clf
                         is the selected classifier's probability for the pair.
Transition             : Newson & Krumm (2009) style. For candidates a (t-1) and b (t):
                           d_route = network distance driven from (a, s_a) to (b, s_b)
                           d_gc    = straight-line distance between the two GPS points
                           log P(b | a) = -|d_route - d_gc| / beta_eff
                         with beta_eff = beta * sqrt(max(dt, 1)) so larger time gaps
                         tolerate larger mismatch. Unreachable pairs get -inf.
Breaks                 : if every transition is -inf (disconnected candidates) or the
                         GPS jump is larger than the search limit, the chain is cut
                         and Viterbi restarts from the emission scores (counted as a break).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass
class HMMConfig:
    beta: float = 8.0          # transition scale (m)
    w_emit: float = 1.0        # emission weight
    back_tol: float = 12.0     # allowed backward jitter on the same arc (m)
    sp_limit: float = 3000.0   # Dijkstra search limit (m)
    min_prob: float = 1e-4


class NetworkDistance:
    """Cached bounded single-source shortest paths over the directed node graph."""

    def __init__(self, trip, limit):
        V = len(trip.nodes)
        u, v = trip.arcs[:, 0], trip.arcs[:, 1]
        w = np.maximum(trip.arc_len, 0.01)
        # keep shortest of parallel arcs (csr would otherwise sum duplicates)
        order = np.lexsort((w, v, u))
        u, v, w = u[order], v[order], w[order]
        first = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
        self.G = csr_matrix((w[first], (u[first], v[first])), shape=(V, V))
        self.limit = limit
        self.cache = {}
        self.arc_of = {}
        for k, (a, b) in enumerate(trip.arcs):
            if (a, b) not in self.arc_of or trip.arc_len[k] < trip.arc_len[self.arc_of[(a, b)]]:
                self.arc_of[(int(a), int(b))] = k

    def _row(self, src):
        r = self.cache.get(src)
        if r is None:
            d = dijkstra(self.G, directed=True, indices=src, limit=self.limit)
            nz = np.flatnonzero(np.isfinite(d))
            r = (nz, d[nz])
            self.cache[src] = r
        return r

    def dist(self, src, dsts):
        nodes, d = self._row(int(src))
        pos = np.searchsorted(nodes, dsts)
        pos = np.clip(pos, 0, len(nodes) - 1)
        hit = nodes[pos] == dsts
        return np.where(hit, d[pos], np.inf)

    def path_arcs(self, src, dst):
        """Arc ids of the shortest path src->dst (for filling route gaps)."""
        d, pred = dijkstra(self.G, directed=True, indices=int(src), limit=self.limit,
                           return_predecessors=True)
        if not np.isfinite(d[dst]):
            return None
        nodes = [int(dst)]
        while nodes[-1] != src:
            p = pred[nodes[-1]]
            if p < 0:
                return None
            nodes.append(int(p))
        nodes = nodes[::-1]
        return [self.arc_of[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1)]


def _group_slices(points_sorted):
    start = np.r_[0, np.flatnonzero(np.diff(points_sorted)) + 1]
    end = np.r_[start[1:], len(points_sorted)]
    return start, end


def viterbi_decode(trip, cand_df, prob, cfg: HMMConfig = HMMConfig(), net=None):
    """cand_df rows sorted by point (columns point, arc, t_proj). prob aligned to rows.
    Returns (pred_arc per point, n_breaks)."""
    net = net or NetworkDistance(trip, cfg.sp_limit)
    pts = cand_df["point"].to_numpy()
    arcs_all = cand_df["arc"].to_numpy()
    s_all = cand_df["t_proj"].to_numpy() * trip.arc_len[arcs_all]
    logp = cfg.w_emit * np.log(np.clip(prob, cfg.min_prob, 1.0))
    st, en = _group_slices(pts)
    point_ids = pts[st]
    n = trip.n_points
    xy, t = trip.xy, trip.t
    A_from, A_to, L = trip.arcs[:, 0], trip.arcs[:, 1], trip.arc_len

    backs, cand_arcs = [], []
    end_scores = {}           # step index -> forward scores at the end of a segment
    delta = None
    breaks = 0
    sp_cache = {}
    prev = None
    for k in range(len(st)):
        sl = slice(st[k], en[k])
        a_cur, s_cur, e_cur = arcs_all[sl], s_all[sl], logp[sl]
        T = None
        if delta is not None:
            a_prev, s_prev, pidx = prev
            gc = float(np.hypot(*(xy[point_ids[k]] - xy[pidx])))
            dt = max(float(t[point_ids[k]] - t[pidx]), 1.0)
            if gc <= cfg.sp_limit * 0.8:
                key = (a_prev.tobytes(), a_cur.tobytes())
                SP = sp_cache.get(key)
                if SP is None:
                    SP = np.vstack([net.dist(A_to[a], A_from[a_cur]) for a in a_prev])
                    if len(sp_cache) > 5000:
                        sp_cache.clear()
                    sp_cache[key] = SP
                route_d = (L[a_prev] - s_prev)[:, None] + SP + s_cur[None, :]
                same = a_prev[:, None] == a_cur[None, :]
                fwd = s_cur[None, :] - s_prev[:, None]
                same_d = np.where(fwd >= -cfg.back_tol, np.maximum(fwd, 0), 3 * np.abs(fwd))
                route_d = np.where(same, same_d, route_d)
                T = -np.abs(route_d - gc) / (cfg.beta * np.sqrt(dt))
                if not np.isfinite(delta[:, None] + T).any():
                    T = None
            if T is None:
                breaks += 1
                end_scores[k - 1] = delta
        if T is None:
            delta = e_cur.astype(float).copy()
            backs.append(None)
        else:
            sc = delta[:, None] + T
            bp = np.argmax(sc, axis=0)
            delta = sc[bp, np.arange(len(a_cur))] + e_cur
            delta = delta - np.max(delta[np.isfinite(delta)])   # avoid numeric drift
            backs.append(bp)
        cand_arcs.append(a_cur)
        prev = (a_cur, s_cur, point_ids[k])
    end_scores[len(st) - 1] = delta

    choice = np.zeros(len(st), dtype=np.int64)
    for k in range(len(st) - 1, -1, -1):
        if k in end_scores:
            choice[k] = int(np.argmax(end_scores[k]))
        else:
            choice[k] = backs[k + 1][choice[k + 1]]
    pred = np.full(n, -1, dtype=np.int64)
    for k in range(len(st)):
        pred[point_ids[k]] = cand_arcs[k][choice[k]]
    return pred, breaks


def fill_route(pred_arcs, net, arcs):
    """Collapse per-point arcs into a route and connect gaps with shortest paths."""
    from .metrics import collapse
    seq = [int(a) for a in collapse(pred_arcs[pred_arcs >= 0])]
    if not seq:
        return []
    out = [seq[0]]
    for b in seq[1:]:
        va, ub = arcs[out[-1], 1], arcs[b, 0]
        if va != ub:
            mid = net.path_arcs(va, ub)
            if mid and len(mid) <= 60:
                out.extend(mid)
        out.append(b)
    return out
