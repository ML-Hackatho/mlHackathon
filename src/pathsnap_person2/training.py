"""Train LightGBM and XGBoost on Person 1's candidate feature exports.

The primary validation objective is mean per-trip observation-level Macro F1.
All folds are split by trip_id. Candidate-miss observations are omitted from
fitting (they have no positive candidate) but retained in validation metrics.
Student-test is evaluated after model selection and is never used for tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from pathsnap_person1.features import FEATURES

SEED = 42
N_JOBS = 4
LOG = logging.getLogger("pathsnap_person2.training")


def _read_candidate_table(path: Path) -> pd.DataFrame:
    dtype: dict[str, Any] = {"trip_id": "string"}
    dtype.update({column: np.float32 for column in FEATURES})
    for column in ("point", "arc", "obs_index", "arc_id", "target_arc_id"):
        dtype[column] = np.int32
    dtype.update({
        "label_available": "bool",
        "is_target": "bool",
        "candidate_miss": "bool",
    })
    table = pd.read_csv(path, compression="gzip", dtype=dtype)
    required = {"trip_id", "point", "arc", "arc_id", "target_arc_id",
                "label_available", "is_target", "candidate_miss", *FEATURES}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if not table["label_available"].all():
        table = table.loc[table["label_available"]].copy()
    matrix = table[FEATURES].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} has non-finite values in model features")
    return table.reset_index(drop=True)


def _make_estimator(name: str, params: dict[str, Any], positive_weight: float):
    if name == "lightgbm":
        settings = {
            "objective": "binary",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "scale_pos_weight": positive_weight,
            "n_jobs": N_JOBS,
            "random_state": SEED,
            "verbosity": -1,
        }
        settings.update(params)
        return lgb.LGBMClassifier(**settings)
    if name == "xgboost":
        settings = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cpu",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 2.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "scale_pos_weight": positive_weight,
            "n_jobs": N_JOBS,
            "random_state": SEED,
            "verbosity": 0,
        }
        settings.update(params)
        return xgb.XGBClassifier(**settings)
    raise ValueError(f"Unsupported model: {name}")


def _positive_weight(y: np.ndarray) -> float:
    positives = int(np.count_nonzero(y == 1))
    negatives = int(np.count_nonzero(y == 0))
    if positives == 0:
        raise ValueError("Training fold has no positive candidate labels")
    return negatives / positives


def _metrics(table: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int]:
    if len(table) != len(probability):
        raise ValueError("Probability vector length does not match candidate rows")
    y = table["is_target"].astype(np.uint8).to_numpy()
    pred = (probability >= 0.5).astype(np.uint8)
    result: dict[str, float | int] = {
        "candidate_rows": int(len(table)),
        "observations": int(table[["trip_id", "point"]].drop_duplicates().shape[0]),
        "candidate_precision": float(precision_score(y, pred, zero_division=0)),
        "candidate_recall": float(recall_score(y, pred, zero_division=0)),
        "candidate_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "candidate_log_loss": float(log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }
    scored = table[["trip_id", "point", "arc_id", "target_arc_id", "candidate_miss"]].copy()
    scored["_probability"] = probability
    best_indices = scored.groupby(["trip_id", "point"], sort=False)["_probability"].idxmax()
    best = scored.loc[best_indices, ["trip_id", "point", "arc_id"]].rename(
        columns={"arc_id": "pred_arc_id"}
    )
    truth = scored.groupby(["trip_id", "point"], sort=False).agg(
        true_arc_id=("target_arc_id", "first"),
        candidate_miss=("candidate_miss", "first"),
    ).reset_index()
    observations = truth.merge(best, on=["trip_id", "point"], how="left", validate="one_to_one")
    observations["correct"] = observations["true_arc_id"] == observations["pred_arc_id"]
    per_trip = [
        f1_score(g["true_arc_id"], g["pred_arc_id"], average="macro", zero_division=0)
        for _, g in observations.groupby("trip_id", sort=False)
    ]
    result.update({
        "observation_top1_accuracy": float(observations["correct"].mean()),
        "observation_macro_f1_mean_per_trip": float(np.mean(per_trip)),
        "candidate_miss_rate": float(observations["candidate_miss"].mean()),
    })
    return result


def _trip_folds(train: pd.DataFrame, n_splits: int):
    trips = np.asarray(sorted(train["trip_id"].astype(str).unique()))
    splitter = GroupKFold(n_splits=min(n_splits, len(trips)))
    return [
        (trips[fit_idx], trips[valid_idx])
        for fit_idx, valid_idx in splitter.split(trips, groups=trips)
    ]


def _cross_validate(
    name: str,
    params: dict[str, Any],
    table: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    include_fit_metrics: bool,
) -> list[dict[str, Any]]:
    X = table[FEATURES].to_numpy(dtype=np.float32, copy=False)
    y = table["is_target"].astype(np.uint8).to_numpy()
    trip = table["trip_id"].astype(str).to_numpy()
    miss = table["candidate_miss"].to_numpy(dtype=bool)
    records: list[dict[str, Any]] = []
    for fold_no, (fit_trips, valid_trips) in enumerate(folds, start=1):
        fit_mask = np.isin(trip, fit_trips) & ~miss
        valid_mask = np.isin(trip, valid_trips)
        fit_idx = np.flatnonzero(fit_mask)
        valid_idx = np.flatnonzero(valid_mask)
        y_fit = y[fit_idx]
        model = _make_estimator(name, params, _positive_weight(y_fit))
        t0 = time.perf_counter()
        model.fit(X[fit_idx], y_fit)
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        valid_prob = model.predict_proba(X[valid_idx])[:, 1].astype(np.float64)
        infer_seconds = time.perf_counter() - t0
        valid_metrics = _metrics(table.iloc[valid_idx], valid_prob)
        record: dict[str, Any] = {
            "model": name,
            "fold": fold_no,
            "fit_trip_count": int(len(fit_trips)),
            "validation_trip_count": int(len(valid_trips)),
            "fit_seconds": float(fit_seconds),
            "validation_predict_seconds": float(infer_seconds),
            **valid_metrics,
        }
        if include_fit_metrics:
            fit_prob = model.predict_proba(X[fit_idx])[:, 1].astype(np.float64)
            fit_metrics = _metrics(table.iloc[fit_idx], fit_prob)
            for key in ("candidate_macro_f1", "observation_top1_accuracy", "observation_macro_f1_mean_per_trip"):
                record[f"train_{key}"] = fit_metrics[key]
        records.append(record)
        LOG.info(
            "%s fold %d/%d: CV observation Macro F1=%.4f Top1=%.4f fit=%.1fs",
            name, fold_no, len(folds), valid_metrics["observation_macro_f1_mean_per_trip"],
            valid_metrics["observation_top1_accuracy"], fit_seconds,
        )
    return records


def _summarize_cv(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    metrics = [
        "candidate_macro_f1", "candidate_log_loss", "observation_top1_accuracy",
        "observation_macro_f1_mean_per_trip", "candidate_miss_rate",
    ]
    out: dict[str, Any] = {"folds": int(len(frame))}
    for metric in metrics:
        out[f"cv_{metric}_mean"] = float(frame[metric].mean())
        out[f"cv_{metric}_std"] = float(frame[metric].std(ddof=1))
    if "train_observation_macro_f1_mean_per_trip" in frame:
        out["cv_train_observation_macro_f1_mean"] = float(frame["train_observation_macro_f1_mean_per_trip"].mean())
        out["cv_train_top1_mean"] = float(frame["train_observation_top1_accuracy"].mean())
        out["cv_train_candidate_macro_f1_mean"] = float(frame["train_candidate_macro_f1"].mean())
        out["cv_observation_macro_f1_train_valid_gap"] = (
            out["cv_train_observation_macro_f1_mean"]
            - out["cv_observation_macro_f1_mean_per_trip_mean"]
        )
    return out


def _suggest_params(trial: optuna.Trial, name: str) -> dict[str, Any]:
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63, step=4),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100, step=10),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def _tune_model(
    name: str,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    n_trials: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    storage = f"sqlite:///{(output_dir / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(
        study_name=f"pathsnap_{name}_grouped_cv",
        storage=storage,
        load_if_exists=False,
        direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=min(3, n_trials)),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, name)
        fold_records = _cross_validate(name, params, train, folds, include_fit_metrics=False)
        fold_scores = [r["observation_macro_f1_mean_per_trip"] for r in fold_records]
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_std", float(np.std(fold_scores, ddof=1)))
        return float(np.mean(fold_scores))

    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    tune_seconds = time.perf_counter() - started
    trials_frame = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    trials_frame.to_csv(output_dir / "optuna_trials.csv", index=False)
    summary = {
        "selected_model_family": name,
        "sampler": "Optuna TPESampler",
        "seed": SEED,
        "objective": "mean per-trip observation-level Macro F1 over five GroupKFold folds (group=trip_id)",
        "n_trials": n_trials,
        "search_space": (
            {
                "n_estimators": "int [200, 500], step 100",
                "max_depth": "int [3, 9]",
                "learning_rate": "log-uniform [0.02, 0.15]",
                "min_child_weight": "log-uniform [1, 20]",
                "subsample": "uniform [0.7, 1.0]",
                "colsample_bytree": "uniform [0.7, 1.0]",
                "reg_lambda": "log-uniform [0.1, 10]",
                "gamma": "uniform [0, 2]",
            } if name == "xgboost" else {
                "n_estimators": "int [200, 500], step 100",
                "num_leaves": "int [15, 63], step 4",
                "learning_rate": "log-uniform [0.02, 0.15]",
                "max_depth": "int [5, 12]",
                "min_child_samples": "int [20, 100], step 10",
                "subsample": "uniform [0.7, 1.0]",
                "colsample_bytree": "uniform [0.7, 1.0]",
                "reg_lambda": "log-uniform [0.1, 10]",
            }
        ),
        "best_value_cv_observation_macro_f1": float(study.best_value),
        "best_params": study.best_params,
        "tuning_seconds": float(tune_seconds),
    }
    (output_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return dict(study.best_params), trials_frame, summary


def _fit_full(name: str, params: dict[str, Any], trainable: pd.DataFrame):
    X = trainable[FEATURES].to_numpy(dtype=np.float32, copy=False)
    y = trainable["is_target"].astype(np.uint8).to_numpy()
    model = _make_estimator(name, params, _positive_weight(y))
    start = time.perf_counter()
    model.fit(X, y)
    return model, float(time.perf_counter() - start)


def _save_json(path: Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            number = float(item)
            return number if math.isfinite(number) else None
        if isinstance(item, np.bool_):
            return bool(item)
        return item

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def train(data_dir: Path, out_dir: Path, folds_n: int = 5, n_trials: int = 5) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    file_train = data_dir / "train_candidates.csv.gz"
    file_student = data_dir / "student_test_candidates.csv.gz"
    if not file_train.is_file() or not file_student.is_file():
        raise FileNotFoundError(
            "Expected train_candidates.csv.gz and student_test_candidates.csv.gz in "
            f"{data_dir}; export them with pathsnap-export-candidates first."
        )
    LOG.info("Reading candidate exports from %s", data_dir)
    train_table = _read_candidate_table(file_train)
    student_table = _read_candidate_table(file_student)
    train_hits = train_table.loc[~train_table["candidate_miss"]].copy().reset_index(drop=True)
    if train_table["trip_id"].nunique() < folds_n:
        raise ValueError(f"Need at least {folds_n} distinct training trips")
    if train_hits["is_target"].nunique() != 2:
        raise ValueError("Training candidate-hit rows must contain both target classes")
    folds = _trip_folds(train_table, folds_n)
    baseline_params: dict[str, dict[str, Any]] = {"lightgbm": {}, "xgboost": {}}
    model_summaries: dict[str, dict[str, Any]] = {}
    fold_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    LOG.info(
        "Loaded %d training rows (%d trips, %d observations); %d hit rows for fitting; "
        "%d student-test rows (%d trips, %d observations)",
        len(train_table), train_table["trip_id"].nunique(), train_table[["trip_id", "point"]].drop_duplicates().shape[0],
        len(train_hits), len(student_table), student_table["trip_id"].nunique(),
        student_table[["trip_id", "point"]].drop_duplicates().shape[0],
    )

    for name in ("lightgbm", "xgboost"):
        LOG.info("Starting five-fold trip-grouped CV for %s", name)
        records = _cross_validate(name, baseline_params[name], train_table, folds, include_fit_metrics=True)
        fold_rows.extend(records)
        cv_summary = _summarize_cv(records)
        model, fit_seconds = _fit_full(name, baseline_params[name], train_hits)
        X_student = student_table[FEATURES].to_numpy(dtype=np.float32, copy=False)
        start = time.perf_counter()
        student_prob = model.predict_proba(X_student)[:, 1].astype(np.float64)
        inference_seconds = float(time.perf_counter() - start)
        student_metrics = _metrics(student_table, student_prob)
        model_summaries[name] = {
            "family": name,
            "initial_parameters": model.get_params(deep=False),
            "training_objective": "binary log loss; positive class is is_target=True",
            "cv": cv_summary,
            "student_test": student_metrics,
            "fit_seconds_full_train": fit_seconds,
            "student_test_predict_seconds": inference_seconds,
            "student_test_predict_ms_per_1000_observations": inference_seconds / max(
                student_metrics["observations"], 1
            ) * 1000.0,
        }
        row = {"model": name, **cv_summary, **{f"student_{k}": v for k, v in student_metrics.items()},
               "fit_seconds_full_train": fit_seconds, "student_test_predict_seconds": inference_seconds}
        comparison_rows.append(row)
        _save_json(models_dir / f"{name}_baseline.json", {
            "model_name": name,
            "model_class": model.__class__.__name__,
            "feature_columns": FEATURES,
            "parameters": model.get_params(deep=False),
            "training_objective": "binary log loss; positive class is is_target=True",
            "random_seed": SEED,
            "training_trips": int(train_hits["trip_id"].nunique()),
            "training_candidate_rows": int(len(train_hits)),
        })

    comparison_cv = pd.DataFrame(comparison_rows)
    selected = str(comparison_cv.sort_values(
        "cv_observation_macro_f1_mean_per_trip_mean", ascending=False
    ).iloc[0]["model"])
    LOG.info("Selected %s for Bayesian tuning by grouped CV observation Macro F1", selected)
    best_params, trials_frame, tuning_summary = _tune_model(
        selected, train_table, folds, out_dir, n_trials
    )
    tuned_fold_rows = _cross_validate(selected, best_params, train_table, folds, include_fit_metrics=True)
    for rec in tuned_fold_rows:
        rec["model"] = f"{selected}_tuned"
    fold_rows.extend(tuned_fold_rows)
    tuned_cv = _summarize_cv(tuned_fold_rows)
    tuned_fit, tuned_fit_seconds = _fit_full(selected, best_params, train_hits)
    X_student = student_table[FEATURES].to_numpy(dtype=np.float32, copy=False)
    start = time.perf_counter()
    tuned_prob = tuned_fit.predict_proba(X_student)[:, 1].astype(np.float64)
    tuned_infer_seconds = float(time.perf_counter() - start)
    tuned_student = _metrics(student_table, tuned_prob)
    comparison_rows.append({
        "model": f"{selected}_tuned",
        **tuned_cv,
        **{f"student_{k}": v for k, v in tuned_student.items()},
        "fit_seconds_full_train": tuned_fit_seconds,
        "student_test_predict_seconds": tuned_infer_seconds,
    })
    model_summaries[f"{selected}_tuned"] = {
        "family": selected,
        "selected_by": "highest mean observation-level Macro F1 among LightGBM and XGBoost baseline GroupKFold runs",
        "best_parameters": best_params,
        "training_objective": "binary log loss; positive class is is_target=True",
        "cv": tuned_cv,
        "student_test": tuned_student,
        "fit_seconds_full_train": tuned_fit_seconds,
        "student_test_predict_seconds": tuned_infer_seconds,
    }
    joblib.dump(tuned_fit, models_dir / "final_model.joblib", compress=3)
    if selected == "xgboost":
        tuned_fit.save_model(str(models_dir / "final_model.ubj"))
    elif selected == "lightgbm":
        tuned_fit.booster_.save_model(str(models_dir / "final_model.txt"))
    final_config = {
        "model_name": selected,
        "model_file": "final_model.joblib",
        "native_model_file": "final_model.ubj" if selected == "xgboost" else "final_model.txt",
        "feature_columns": FEATURES,
        "positive_class": "is_target=True",
        "parameters": tuned_fit.get_params(deep=False),
        "training_objective": "binary log loss",
        "seed": SEED,
        "selected_by": "mean per-trip observation Macro F1 on five-fold GroupKFold by trip_id",
        "tuning": tuning_summary,
        "training_rows": int(len(train_hits)),
        "training_trips": int(train_hits["trip_id"].nunique()),
        "candidate_miss_training_observations_excluded": int(
            train_table.groupby(["trip_id", "point"], sort=False)["candidate_miss"].first().sum()
        ),
    }
    _save_json(models_dir / "final_model.json", final_config)
    pd.DataFrame(fold_rows).to_csv(out_dir / "grouped_cv_folds.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "model_comparison_results.csv", index=False)
    model_summaries["selected_model"] = selected
    model_summaries["data"] = {
        "training_candidate_rows": int(len(train_table)),
        "training_observations": int(train_table[["trip_id", "point"]].drop_duplicates().shape[0]),
        "training_trips": int(train_table["trip_id"].nunique()),
        "training_candidate_hit_rows": int(len(train_hits)),
        "training_candidate_miss_observations": int(
            train_table.groupby(["trip_id", "point"], sort=False)["candidate_miss"].first().sum()
        ),
        "student_test_candidate_rows": int(len(student_table)),
        "student_test_observations": int(student_table[["trip_id", "point"]].drop_duplicates().shape[0]),
        "student_test_trips": int(student_table["trip_id"].nunique()),
        "feature_columns": FEATURES,
        "folds": folds_n,
        "group_column": "trip_id",
        "candidate_miss_handling": "excluded from fit; retained in validation and counted as Top-1 errors",
        "evaluator_data_used": False,
    }
    model_summaries["software"] = {
        "python": __import__("sys").version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": xgb.__version__,
        "optuna": optuna.__version__,
    }
    _save_json(out_dir / "training_summary.json", model_summaries)
    _plot_comparison(comparison, out_dir / "model_comparison.png")
    LOG.info("Wrote model results, tuning trials, configurations and final model to %s", out_dir)
    return comparison


def _plot_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    rows = comparison.set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    cv_keys = [c for c in rows.index]
    means = [rows.loc[k, "cv_observation_macro_f1_mean_per_trip_mean"] for k in cv_keys]
    stds = [rows.loc[k, "cv_observation_macro_f1_mean_per_trip_std"] for k in cv_keys]
    axes[0].bar(cv_keys, means, yerr=stds, capsize=4, color=["#5a8dee", "#41b883", "#f0a23b"][:len(cv_keys)])
    axes[0].set_title("Grouped CV: observation Macro F1")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)
    for col, label in (("student_observation_top1_accuracy", "Top-1 accuracy"),
                       ("student_observation_macro_f1_mean_per_trip", "Macro F1")):
        axes[1].plot(cv_keys, [rows.loc[k, col] for k in cv_keys], marker="o", label=label)
    axes[1].set_title("Student-test observation scores")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    axes[2].bar(cv_keys, [rows.loc[k, "student_candidate_log_loss"] for k in cv_keys], color="#9575cd")
    axes[2].set_title("Student-test candidate Log Loss")
    axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle("LightGBM and XGBoost - actual grouped-CV and student-test results")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare and tune PathSnap LightGBM/XGBoost models.")
    parser.add_argument("--candidate-dir", type=Path, default=Path("outputs/shivanand_handoff"))
    parser.add_argument("--out", type=Path, default=Path("outputs/person2"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "training.log", encoding="utf-8")],
    )
    result = train(args.candidate_dir, args.out, args.folds, args.trials)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
