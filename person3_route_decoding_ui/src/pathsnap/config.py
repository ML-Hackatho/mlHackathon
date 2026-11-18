"""Paths and global settings (Person 3: route decoding + UI).

This folder lives inside the team repository:
    mlHackathon/
      src/pathsnap_person1/           <- Person 1: loader, candidates, features, labels
      outputs/shivanand_handoff/      <- Person 1: exported candidate tables (train / student_test)
      person3_route_decoding_ui/      <- this folder
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAM_ROOT = os.path.dirname(ROOT)
PERSON1_SRC = os.path.join(TEAM_ROOT, "src")
HANDOFF = os.path.join(TEAM_ROOT, "outputs", "shivanand_handoff")
if PERSON1_SRC not in sys.path:
    sys.path.insert(0, PERSON1_SRC)

DATA = os.path.join(ROOT, "data")
TRAIN_DIR = os.path.join(DATA, "train")
STUDENT_TEST_DIR = os.path.join(DATA, "student_test")
PROCESSED = os.path.join(DATA, "processed")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
MAPS = os.path.join(ROOT, "maps")

SEED = 42
N_FOLDS = 5
TRAIN_POINT_STRIDE = 3   # use every 3rd GPS point (with all its candidates) for fitting

for d in (PROCESSED, MODELS, RESULTS, FIGURES, MAPS):
    os.makedirs(d, exist_ok=True)
