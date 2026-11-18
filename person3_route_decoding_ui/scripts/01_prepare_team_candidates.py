"""Load Person 1's exported candidate tables (the team's shared features and labels).

Input  : ../outputs/shivanand_handoff/{train,student_test}_candidates.csv.gz   (Person 1)
Output : data/processed/{train,student_test}_candidates.parquet
         results/candidate_tables_summary.csv

Columns added for this folder's scripts (no values are changed):
  trip     = trip_id (kept as a zero-padded string)
  y        = is_target as 0/1   (target: candidate arc == aligned reference arc)
  reliable = label_available    (every labelled point is scored; candidate-miss points
                                 count as errors, following Person 1's evaluation rule)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.team import FEATURES  # noqa: E402


def prepare(name):
    path = os.path.join(C.HANDOFF, f"{name}_candidates.csv.gz")
    df = pd.read_csv(path, compression="gzip", dtype={"trip_id": str})
    missing = [c for c in ["point", "arc", "t_proj", "is_target", "candidate_miss", *FEATURES] if c not in df]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}")
    df["trip"] = df["trip_id"]
    df["y"] = df["is_target"].astype(bool).astype(int)
    df["reliable"] = df["label_available"].astype(bool)
    df["candidate_miss"] = df["candidate_miss"].astype(bool)
    df = df.sort_values(["trip", "point"], kind="stable").reset_index(drop=True)
    df.to_parquet(os.path.join(C.PROCESSED, f"{name}_candidates.parquet"))
    pts = df.drop_duplicates(["trip", "point"])
    s = {"partition": name, "trips": df["trip"].nunique(), "points": len(pts), "rows": len(df),
         "mean_candidates": len(df) / len(pts), "candidate_miss_rate": pts["candidate_miss"].mean(),
         "positive_rate": df["y"].mean()}
    print(s)
    return s


if __name__ == "__main__":
    s = pd.DataFrame([prepare("train"), prepare("student_test")])
    s.to_csv(os.path.join(C.RESULTS, "candidate_tables_summary.csv"), index=False)
