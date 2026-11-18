"""Evaluation metrics (candidate-level, observation-level, sequence-level)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "macro_precision": precision_score(y, yhat, average="macro", zero_division=0),
        "macro_recall": recall_score(y, yhat, average="macro", zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_from_scores(df, score_col="p"):
    """Highest-scoring candidate arc per point -> pd.Series(point -> arc)."""
    idx = df.groupby(["trip", "point"])[score_col].idxmax()
    best = df.loc[idx, ["trip", "point", "arc"]]
    return best


def top1_accuracy(df, score_col="p"):
    """Observation-level Top-1 accuracy over reliable points. A point whose
    reference arc is absent from its candidate set counts as an error."""
    rel = df[df["reliable"]]
    best = rel.loc[rel.groupby(["trip", "point"])[score_col].idxmax()]
    return float(best["y"].mean())


def candidate_miss_rate(df):
    rel = df[df["reliable"]]
    has = rel.groupby(["trip", "point"])["y"].max()
    return float(1 - has.mean())


def collapse(seq):
    seq = np.asarray(seq)
    if len(seq) == 0:
        return seq
    keep = np.r_[True, seq[1:] != seq[:-1]]
    return seq[keep]


def route_continuity(arc_seq, arcs):
    """Fraction of consecutive (distinct) predicted arcs that are directly
    connected in the directed network (to-node of a == from-node of b)."""
    c = collapse(arc_seq)
    if len(c) < 2:
        return 1.0
    ok = arcs[c[:-1], 1] == arcs[c[1:], 0]
    return float(ok.mean())


def route_overlap(pred_arcs, route, arc_len):
    """Sequence-level comparison of the predicted arc set with the reference route.
    Returns arc-count and length-weighted precision / recall / F1."""
    P, R = set(np.unique(pred_arcs).tolist()), set(np.unique(route).tolist())
    inter = P & R
    out = {}
    for name, w in (("", None), ("len_", arc_len)):
        if w is None:
            tp, pp, rr = len(inter), len(P), len(R)
        else:
            tp = w[list(inter)].sum() if inter else 0.0
            pp = w[list(P)].sum() if P else 0.0
            rr = w[list(R)].sum() if R else 0.0
        prec = tp / pp if pp else 0.0
        rec = tp / rr if rr else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.update({f"route_{name}precision": prec, f"route_{name}recall": rec, f"route_{name}f1": f1})
    return out


def pointwise_accuracy(pred_arc, labels):
    m = labels["reliable"]
    return float((pred_arc[m] == labels["label_arc"][m]).mean())
