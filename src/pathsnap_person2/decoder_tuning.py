"""Tune Shreya's HMM for the selected model using training-only OOF scores."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from pathsnap_person1.geometry import build_road_index
from pathsnap_person1.io import discover_trip_dirs, load_trip

from .decoder import HMMConfig, NetworkDistance, collapse_sequence, viterbi_decode
from .inference import DecoderTrip, RoutePredictor, _continuity, _route_overlap, score_against_reference
from .oof import export_oof
from .training import _read_candidate_table

LOG = logging.getLogger("pathsnap_person2.decoder_tuning")


@dataclass
class PreparedTrip:
    trip_id: str
    decoder_trip: DecoderTrip
    candidate_rows: pd.DataFrame
    probability: np.ndarray
    target_arc: np.ndarray
    classifier_only: np.ndarray
    network: NetworkDistance


def _prepare_training_trips(
    table: pd.DataFrame,
    probability: np.ndarray,
    data_root: Path,
    sp_limit: float,
) -> list[PreparedTrip]:
    trip_dirs = {p.name: p for p in discover_trip_dirs(data_root, "train")}
    trip_column = table["trip_id"].astype(str).to_numpy()
    prepared: list[PreparedTrip] = []
    for trip_id in sorted(np.unique(trip_column)):
        if trip_id not in trip_dirs:
            raise FileNotFoundError(f"No raw train trip directory for candidate export trip {trip_id}")
        row_idx = np.flatnonzero(trip_column == trip_id)
        candidates = table.iloc[row_idx]
        trip = load_trip(trip_dirs[trip_id], "train", include_route=False)
        index = build_road_index(trip)
        decoder_trip = DecoderTrip(
            trip_id=trip.trip_id,
            track=trip.track,
            nodes=trip.nodes,
            arcs=trip.arcs,
            xy=index.projected_gps,
            arc_len=index.lengths_m,
        )
        target = np.full(len(trip.track), -1, dtype=np.int64)
        first = candidates.drop_duplicates("point")
        target[first["point"].to_numpy(dtype=np.int64)] = first["target_arc_id"].to_numpy(dtype=np.int64)
        p = probability[row_idx].astype(np.float64, copy=False)
        scored = candidates[["point", "arc", "t_proj"]].copy()
        scored["probability"] = p
        best = scored.loc[scored.groupby("point", sort=False)["probability"].idxmax()]
        classifier = np.full(len(trip.track), -1, dtype=np.int64)
        classifier[best["point"].to_numpy(dtype=np.int64)] = best["arc"].to_numpy(dtype=np.int64)
        net = NetworkDistance(decoder_trip, sp_limit)
        # Gap-filling is not part of the OOF decoder objective; retaining the
        # arc lookup dictionary for all 80 trip graphs would waste memory.
        net.arc_of.clear()
        prepared.append(PreparedTrip(trip_id, decoder_trip, scored, p, target, classifier, net))
    return prepared


def _decode_scores(prepared: list[PreparedTrip], config: HMMConfig) -> tuple[float, list[dict[str, Any]]]:
    trip_records: list[dict[str, Any]] = []
    for item in prepared:
        decoded, breaks = viterbi_decode(
            item.decoder_trip,
            item.candidate_rows[["point", "arc", "t_proj"]],
            item.probability,
            config,
            item.network,
        )
        reliable = item.target_arc >= 0
        accuracy = float(np.mean(decoded[reliable] == item.target_arc[reliable])) if reliable.any() else np.nan
        trip_records.append({
            "trip_id": item.trip_id,
            "points": int(reliable.sum()),
            "accuracy_hmm": accuracy,
            "continuity_hmm": _continuity(decoded, item.decoder_trip.arcs),
            "chain_breaks": int(breaks),
        })
        # Rebuild bounded shortest-path rows on the next trial rather than
        # retaining route-sized caches for every training trip simultaneously.
        item.network.cache.clear()
    objective = float(np.nanmean([r["accuracy_hmm"] for r in trip_records]))
    return objective, trip_records


def tune_decoder(
    candidate_dir: Path,
    data_root: Path,
    output_dir: Path,
    n_trials: int = 30,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    config_path = models_dir / "final_model.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run model training first: {config_path}")
    model_info = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = model_info["model_name"]
    oof_path = output_dir / f"oof_{model_name}.npy"
    if not oof_path.is_file():
        LOG.info("Creating row-aligned OOF probabilities for %s", model_name)
        export_oof(candidate_dir, output_dir, folds_n=5)
    table = _read_candidate_table(candidate_dir / "train_candidates.csv.gz")
    probability = np.load(oof_path, mmap_mode="r")
    if len(probability) != len(table):
        raise ValueError("OOF probability count does not match candidate training CSV rows")
    prepared = _prepare_training_trips(table, probability, data_root, sp_limit=3000.0)
    classifier_accuracy = float(np.mean([
        np.mean(item.classifier_only[item.target_arc >= 0] == item.target_arc[item.target_arc >= 0])
        for item in prepared
    ]))
    default_config = HMMConfig()
    default_accuracy, default_records = _decode_scores(prepared, default_config)

    database = output_dir / "decoder_optuna_study.db"
    study = optuna.create_study(
        study_name=f"pathsnap_{model_name}_decoder",
        storage=f"sqlite:///{database.as_posix()}",
        load_if_exists=False,
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=min(10, n_trials)),
    )

    def objective(trial: optuna.Trial) -> float:
        config = HMMConfig(
            beta=trial.suggest_float("beta", 0.5, 64.0, log=True),
            w_emit=1.0,
            back_tol=trial.suggest_float("back_tol", 0.0, 40.0),
            sp_limit=3000.0,
            min_prob=1e-4,
        )
        score, records = _decode_scores(prepared, config)
        trial.set_user_attr("mean_continuity", float(np.mean([r["continuity_hmm"] for r in records])))
        trial.set_user_attr("mean_chain_breaks", float(np.mean([r["chain_breaks"] for r in records])))
        return score

    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    tune_seconds = time.perf_counter() - start
    best_config = HMMConfig(
        beta=float(study.best_params["beta"]),
        w_emit=1.0,
        back_tol=float(study.best_params["back_tol"]),
        sp_limit=3000.0,
        min_prob=1e-4,
    )
    tuned_accuracy, tuned_records = _decode_scores(prepared, best_config)
    trial_frame = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    trial_frame.to_csv(output_dir / "decoder_optuna_trials.csv", index=False)
    pd.DataFrame(default_records).to_csv(output_dir / "decoder_default_train_oof_by_trip.csv", index=False)
    pd.DataFrame(tuned_records).to_csv(output_dir / "decoder_tuned_train_oof_by_trip.csv", index=False)
    config = {
        "model_name": model_name,
        "selected_by": "Optuna TPE on the selected model's five-fold trip-grouped OOF probabilities",
        "objective": "mean decoded observation-level accuracy over 80 training trips",
        "n_trials": int(n_trials),
        "seed": 42,
        "search_space": {"beta_m": "log-uniform [0.5, 64]", "back_tol_m": "uniform [0, 40]"},
        "w_emit": "fixed at 1.0; only the ratio relative to beta changes the path score",
        "oof_probability_file": oof_path.name,
        "training_seconds": float(tune_seconds),
        "classifier_only_train_oof_accuracy": classifier_accuracy,
        "default_hmm_train_oof_accuracy": default_accuracy,
        "tuned_hmm_train_oof_accuracy": tuned_accuracy,
        "hmm": asdict(best_config),
    }
    (models_dir / "decoder_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"setting": "classifier_only", "accuracy": classifier_accuracy},
        {"setting": "default_hmm", "accuracy": default_accuracy, **asdict(default_config)},
        {"setting": "tuned_hmm", "accuracy": tuned_accuracy, **asdict(best_config)},
    ]).to_csv(output_dir / "decoder_train_oof_comparison.csv", index=False)
    (output_dir / "decoder_tuning_summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _plot_decoder_trials(trial_frame, output_dir / "decoder_tuning_history.png")
    LOG.info("HMM tuning complete. OOF accuracy %.4f -> %.4f; config saved to %s",
             default_accuracy, tuned_accuracy, models_dir / "decoder_config.json")
    return config


def _plot_decoder_trials(trials: pd.DataFrame, out_path: Path) -> None:
    completed = trials[trials["state"] == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(completed["number"], completed["value"], marker="o", linewidth=1.2)
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Mean train OOF decoded accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Decoder Bayesian optimisation - actual training OOF trials")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_student_test(candidate_dir: Path, data_root: Path, output_dir: Path) -> pd.DataFrame:
    models_dir = output_dir / "models"
    predictor = RoutePredictor(models_dir / "final_model.joblib", models_dir / "decoder_config.json")
    trip_dirs = discover_trip_dirs(data_root, "student_test")
    rows: list[dict[str, Any]] = []
    for trip_dir in trip_dirs:
        out, trip = predictor.predict(trip_dir, partition="student_test", include_network=False)
        evaluation = score_against_reference(trip_dir, out, "student_test")
        if evaluation is None:
            continue
        # Scoring details are aggregate only; no point labels or references are exported.
        route = load_trip(trip_dir, "student_test", include_route=True).route
        arc_lengths = build_road_index(load_trip(trip_dir, "student_test", include_route=False)).lengths_m
        clf_route = collapse_sequence(np.asarray(out["classifier_only_arcs_per_point"], dtype=np.int64))
        clf_overlap = _route_overlap(clf_route, route, arc_lengths)
        rows.append({
            "trip_id": trip.trip_id,
            "points": int(out["n_points"]),
            "accuracy_classifier_only": evaluation["accuracy_classifier_only"],
            "accuracy_hmm": evaluation["accuracy_hmm"],
            "continuity_classifier_only": out["decoder"]["route_continuity_classifier_only"],
            "continuity_hmm": out["decoder"]["route_continuity_hmm"],
            "route_f1_classifier_only": clf_overlap["route_f1"],
            "route_f1_hmm": evaluation["route_f1"],
            "route_length_f1_hmm": evaluation["route_length_f1"],
            "chain_breaks": out["decoder"]["chain_breaks"],
            "gps_gaps_over_5s": out["decoder"]["gps_gaps_over_5s"],
            "features_ms": out["timing_ms"]["features"],
            "classifier_ms": out["timing_ms"]["classifier"],
            "viterbi_ms": out["timing_ms"]["viterbi"],
            "total_ms": out["timing_ms"]["total"],
            **{f"eval_{key}": value for key, value in evaluation.items()},
        })
        LOG.info("student_test/%s classifier acc %.3f -> HMM %.3f; continuity %.3f -> %.3f",
                 trip.trip_id, evaluation["accuracy_classifier_only"], evaluation["accuracy_hmm"],
                 out["decoder"]["route_continuity_classifier_only"], out["decoder"]["route_continuity_hmm"])
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "hmm_vs_classifier_student_test_by_trip.csv", index=False)
    if not frame.empty:
        summary = frame.drop(columns=["trip_id"]).mean(numeric_only=True).to_dict()
        (output_dir / "hmm_vs_classifier_student_test_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        _plot_hmm_comparison(frame, output_dir / "hmm_vs_classifier_student_test.png")
    return frame


def _plot_hmm_comparison(frame: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("accuracy_classifier_only", "accuracy_hmm", "Observation accuracy"),
        ("continuity_classifier_only", "continuity_hmm", "Route continuity"),
        ("route_f1_classifier_only", "route_f1_hmm", "Route F1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (left, right, title) in zip(axes, metrics):
        ax.bar([0, 1], [frame[left].mean(), frame[right].mean()], color=["#9ecae1", "#08519c"])
        ax.set_xticks([0, 1], ["Classifier", "Classifier + HMM"])
        ax.set_ylim(0, 1)
        ax.set_title(title)
    fig.suptitle("XGBoost/LightGBM selected model: student-test HMM impact")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune decoder on final-model training OOF scores and evaluate student-test.")
    parser.add_argument("--candidate-dir", type=Path, default=Path("outputs/shivanand_handoff"))
    parser.add_argument("--data-root", type=Path, default=Path("data/Student folder"))
    parser.add_argument("--out", type=Path, default=Path("outputs/person2"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--skip-student-eval", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "decoder_tuning.log", encoding="utf-8")],
    )
    tune_decoder(args.candidate_dir, args.data_root, args.out, args.trials)
    if not args.skip_student_eval:
        evaluate_student_test(args.candidate_dir, args.data_root, args.out)


if __name__ == "__main__":
    main()
