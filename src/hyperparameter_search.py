import joblib
from joblib import parallel_config

import warnings
import numpy as np
import pandas as pd
from pprint import pprint

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GroupKFold, RandomizedSearchCV, ParameterGrid, cross_val_predict
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, precision_recall_curve,
)
from xgboost import XGBClassifier

def pr_auc_scorer(estimator, X, y):
    probs = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, probs)


def _tune_threshold_on_train_oof(clf, X_train, y_train, groups_train):
    inner_cv = GroupKFold(n_splits=min(3, len(groups_train.unique())))
    
    try:
        oof_probs = cross_val_predict(
            clf, X_train, y_train, groups=groups_train, cv=inner_cv, method="predict_proba"
        )[:, 1]
    except Exception:
        clf.fit(X_train, y_train)
        oof_probs = clf.predict_proba(X_train)[:, 1]
        
    precision, recall, thresholds = precision_recall_curve(y_train, oof_probs)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    if len(thresholds) == 0:
        return 0.5
        
    best_idx = np.argmax(f1s[:-1])
    return thresholds[best_idx]

def _filter_lr_params(params: dict) -> dict:
    p = dict(params)
    if p.get("penalty") != "elasticnet":
        p.pop("l1_ratio", None)
    return p

# 1. Logistic Regression — Exhaustive Grid
LR_GRID = {
    "C":        [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    "solver":   ["saga"],
}

def search_logistic_regression(X, y, groups, n_splits=5, verbose=True):
    gkf = GroupKFold(n_splits=n_splits)
    best_score, best_params = -np.inf, {}
    rows = []

    total = len(list(ParameterGrid(LR_GRID)))
    for i, params in enumerate(ParameterGrid(LR_GRID)):
        print(f"  [{i+1}/{total}] Testing params: {_filter_lr_params(params)}", end="\r")
        filtered = _filter_lr_params(params)
        
        clf = Pipeline([
            ("scaler", RobustScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced", max_iter=10000, random_state=42, **params
            ))
        ])
        
        fold_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_scores.append(pr_auc_scorer(clf, X.iloc[val_idx], y.iloc[val_idx]))

        mean_pr_auc = np.mean(fold_scores)
        std_pr_auc  = np.std(fold_scores)
        rows.append({**filtered, "mean_pr_auc": mean_pr_auc, "std_pr_auc": std_pr_auc})

        if mean_pr_auc > best_score:
            best_score, best_params = mean_pr_auc, filtered

    cv_df = pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LogisticRegression  —  {len(rows)} combos (exhaustive grid)")
        print(f"{'='*60}")
        print(f"  Best CV PR-AUC : {best_score:.4f}")
        print(f"  Best params:")
        pprint(best_params, indent=6)
        print(f"\n  Top-10:")
        print(cv_df.head(10).to_string(index=False))

    return {"best_params": best_params, "best_score": best_score, "cv_results_df": cv_df}

# 2. Random Forest
RF_N_TRIALS = 150

def _rf_objective(trial, X, y, groups, gkf):
    params = {
        "n_estimators":          trial.suggest_int("n_estimators", 50, 600),
        "max_depth":             trial.suggest_int("max_depth", 2, 12),
        "min_samples_leaf":      trial.suggest_int("min_samples_leaf", 1, 30),
        "min_samples_split":     trial.suggest_int("min_samples_split", 2, 20),
        "max_features":          trial.suggest_categorical(
                                     "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.02),
        "bootstrap":             trial.suggest_categorical("bootstrap", [True, False]),
    }

    clf = RandomForestClassifier(
        class_weight="balanced_subsample", random_state=42, n_jobs=-1, **params
    )

    fold_scores = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_scores.append(pr_auc_scorer(clf, X.iloc[val_idx], y.iloc[val_idx]))

    return float(np.mean(fold_scores))


def search_random_forest(X, y, groups, n_splits=5, n_trials=RF_N_TRIALS, verbose=True):
    gkf = GroupKFold(n_splits=n_splits)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=max(20, n_trials // 10)),
    )
    study.optimize(
        lambda trial: _rf_objective(trial, X, y, groups, gkf),
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    best_params = study.best_params
    best_score  = study.best_value

    rows = [
        {**t.params, "mean_pr_auc": t.value}
        for t in study.trials if t.value is not None
    ]
    cv_df = pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  RandomForest  —  {n_trials} Optuna TPE trials (Bayesian)")
        print(f"{'='*60}")
        print(f"  Best CV PR-AUC : {best_score:.4f}")
        print(f"  Best params:")
        pprint(best_params, indent=6)
        print(f"\n  Top-10 trials:")
        print(cv_df.head(10).to_string(index=False))

    return {"best_params": best_params, "best_score": best_score, "cv_results_df": cv_df}


# 3. XGBoost — Optuna Bayesian optimisation (TPE sampler)

XGB_N_TRIALS = 150


def _xgb_objective(trial, X, y, groups, gkf, scale_pos_weight):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 600),
        "max_depth":         trial.suggest_int("max_depth", 2, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.4, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "gamma":             trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
    }

    clf = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **params,
    )

    fold_scores = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_scores.append(pr_auc_scorer(clf, X.iloc[val_idx], y.iloc[val_idx]))

    return float(np.mean(fold_scores))


def search_xgboost(X, y, groups, n_splits=5, n_trials=XGB_N_TRIALS, verbose=True):
    gkf = GroupKFold(n_splits=n_splits)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=max(20, n_trials // 10)),
    )
    study.optimize(
        lambda trial: _xgb_objective(trial, X, y, groups, gkf, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    best_params = study.best_params
    best_score  = study.best_value

    rows = [
        {**t.params, "mean_pr_auc": t.value}
        for t in study.trials if t.value is not None
    ]
    cv_df = pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  XGBoost  —  {n_trials} Optuna TPE trials (Bayesian)")
        print(f"{'='*60}")
        print(f"  Best CV PR-AUC : {best_score:.4f}")
        print(f"  Best params:")
        pprint(best_params, indent=6)
        print(f"\n  Top-10 trials:")
        print(cv_df.head(10).to_string(index=False))

    return {"best_params": best_params, "best_score": best_score, "cv_results_df": cv_df}

def run_hyperparam_search(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    rf_n_trials: int = RF_N_TRIALS,   # renamed from rf_n_iter
    xgb_n_trials: int = XGB_N_TRIALS,
    verbose: bool = True,
) -> dict:
    results = {}
    #results["LogisticRegression"] = search_logistic_regression(X, y, groups, n_splits, verbose)
    results["RandomForest"]       = search_random_forest(X, y, groups, n_splits, rf_n_trials, verbose)
    results["XGBoost"]            = search_xgboost(X, y, groups, n_splits, xgb_n_trials, verbose)
    return results

def evaluate_best(
    results: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 4,
) -> None:
    gkf = GroupKFold(n_splits=n_splits)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    for clf_name, info in results.items():
        params = info["best_params"]
        print(f"\n{'='*60}")
        print(f"  FINAL EVAL — {clf_name}  (best CV PR-AUC: {info['best_score']:.4f})")
        print(f"{'='*60}")

        if clf_name == "LogisticRegression":
            clf = Pipeline([
                ("scaler", RobustScaler()),
                ("lr", LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=42, **params
                ))
            ])
        elif clf_name == "RandomForest":
            clf = RandomForestClassifier(
                class_weight="balanced_subsample", random_state=42, n_jobs=-1, **params
            )
        else:
            clf = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **params,
            )

        all_rows = []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups_train = groups.iloc[train_idx]

            clf.fit(X_train, y_train)

            best_thresh = _tune_threshold_on_train_oof(clf, X_train, y_train, groups_train)
            preds = (clf.predict_proba(X_test)[:, 1] >= best_thresh).astype(int)

            print(f"  Fold {fold+1} | threshold={best_thresh:.3f} | F1={f1_score(y_test, preds, zero_division=0):.3f}")

            fold_df = X_test.copy()
            fold_df["Model"]      = groups.iloc[test_idx].values
            fold_df["True_Label"] = y_test.values
            fold_df["Prediction"] = preds
            all_rows.append(fold_df)

        all_df = pd.concat(all_rows, ignore_index=True)
        y_true, y_pred = all_df["True_Label"], all_df["Prediction"]

        print(f"\n  Overall Aggregated Metrics:")
        print(f"  Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  F1        : {f1_score(y_true, y_pred, zero_division=0):.4f}")
        print(classification_report(y_true, y_pred))