import ast
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    average_precision_score,
    log_loss
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from xgboost import XGBClassifier

import shap

import hyperparameter_search

def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, "1" if default else "0") == "1"

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))

RUN_HYPERPARAM_SEARCH = _env_bool("RUN_HYPERPARAM_SEARCH", False)
DRY_RUN               = _env_bool("DRY_RUN",               False)
USE_FBETA_THRESHOLD   = _env_bool("USE_FBETA_THRESHOLD",   False)

CV_SPLITS   = _env_int(  "CV_SPLITS",   6)
RF_N_ITER   = _env_int(  "RF_N_ITER",   200)
XGB_N_TRIALS= _env_int(  "XGB_N_TRIALS",150)
THRESHOLD   = _env_float("THRESHOLD",   0.75)
FBETA_BETA  = _env_float("FBETA_BETA",  1.5)

DISTALS_PATH    = os.environ.get("DISTALS_PATH",     "../data/distals/distals-db.pickle")
LANG_SUBSET_PATH= os.environ.get("LANG_SUBSET_PATH", "../data/lang_subset.txt")
OUTPUT_DIR      = Path(os.environ.get("OUTPUT_DIR",  "../data/predictions"))




FEATURE_COLS = [
    "Avg_Tok_Length", "Avg_Tok_Ratio", "Unk_Ratio",
    "whitespace", "punctuation",
    "BPC",
]

BPC_COLS = ["BPC", "BPC25", "BPC50", "BPC75"]

MODEL_TYPE_MAP = {"encoder": 0, "decoder": 1}

def read_size_data(model_path: Path) -> dict:
    lines = (model_path / "size.txt").read_text().strip().splitlines()
    return {
        "Model":      model_path.name,
        "Model_Type": lines[0].strip(),
        "Model_Size": int(lines[1].strip()),
    }


def read_tok_data(path: Path) -> pd.DataFrame:
    present = 1 if "True" in path.name else 0
    model   = path.parent.name
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "Model":          model,
                "Language":       d.get("language"),
                "Unk_Ratio":      d.get("unk ratio"),
                "Avg_Tok_Length": d.get("token_length"),
                "Avg_Tok_Ratio":  d.get("token/word"),
                "In_Data":        present,
            })
    return pd.DataFrame(rows)


def read_bpc_data(path: Path) -> pd.DataFrame:
    present = 1 if "True" in path.name else 0
    model   = path.parent.name
    rows = []
    with open(path) as f:
        for line in f:
            parsed = json.loads(line)
            lang, data = next(iter(parsed.items()))
            rows.append({
                "Model":    model,
                "Language": lang,
                "BPC":      data.get("avg_bpc"),
                "BPC25":    data.get("0.25q"),
                "BPC50":    data.get("0.5q"),
                "BPC75":    data.get("0.75q"),
                "In_Data":  present,
            })
    return pd.DataFrame(rows)


def min_max_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo)


def attach_distals(df: pd.DataFrame, distals: dict) -> pd.DataFrame:
    iso = df["Language"].str[:3]

    scalar_fields = ["whitespace", "punctuation", "speakers", "wiki_size"]
    for field in scalar_fields:
        df[field] = iso.map(lambda x, f=field: distals.get(x, {}).get(f))

    df["nlp_state"] = iso.map(
        lambda x: int(distals.get(x, {}).get("nlp_state", [0])[0])
    )
    df["AES"] = iso.map(
        lambda x: int(distals.get(x, {}).get("AES", [0])[0])
    )

    for col in ["speakers", "wiki_size"]:
        df[col] = min_max_normalize(df[col])

    return df


def balance_classes(df: pd.DataFrame, label_col: str = "In_Data",
                    ratio: dict[int, int] | None = None,
                    random_state: int = 42) -> pd.DataFrame:

    if ratio is None:
        ratio = {0: 1, 1: 1}
    max_multiplier = min(
        (df[label_col] == cls).sum() // weight
        for cls, weight in ratio.items()
    )
    return pd.concat([
        df[df[label_col] == cls].sample(
            n=int(weight * max_multiplier), random_state=random_state
        )
        for cls, weight in ratio.items()
    ]).reset_index(drop=True)


def normalize_per_model(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    def _robust(s: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            RobustScaler().fit_transform(s),
            columns=s.columns,
            index=s.index,
        )
    df = df.copy()
    df[cols] = df.groupby("Model")[cols].apply(_robust).reset_index(level=0, drop=True)
    return df


# load data
DATA_DIR = Path("../data")

with open(DISTALS_PATH, "rb") as f:
    distals_data = pickle.load(f)

# Tokenizer data
dfs_toks = []
for folder in (DATA_DIR / "tokenizer_based").iterdir():
    if folder.is_dir():
        for model_path in folder.iterdir():
            dfs_toks.extend([
                read_tok_data(model_path / "True.json"),
                read_tok_data(model_path / "False.json"),
            ])
df_tok = pd.concat(dfs_toks, ignore_index=True)

# BPC + size data
dfs_bpc, sizes = [], []
for folder in (DATA_DIR / "lang_lengths").iterdir():
    if "DS_Store" in str(folder) or not folder.is_dir():
        continue
    for model_path in folder.iterdir():
        if "DS_Store" in str(model_path):
            continue
        dfs_bpc.extend([
            read_bpc_data(model_path / "0.05_5_True.json"),
            read_bpc_data(model_path / "0.05_5_False.json"),
        ])
        sizes.append(read_size_data(model_path))

df_bpc  = pd.concat(dfs_bpc, ignore_index=True)
size_df = pd.DataFrame(sizes)

# Merge
df = (
    df_tok
    .merge(df_bpc[["Model", "Language", "BPC", "BPC25", "BPC50", "BPC75"]],
           on=["Model", "Language"], how="inner")
    .merge(size_df, on="Model", how="left")
)

df["Model_Type"] = df["Model_Type"].map(MODEL_TYPE_MAP)
df["Model_Size"]  = min_max_normalize(df["Model_Size"])
df["Script"]      = df["Language"].str.split("_").str[1]

le = LabelEncoder()
df["Script_id"] = le.fit_transform(df["Script"])

model_counts = df.groupby("Model")["In_Data"].sum()
df["Model_In_Data_Count"] = df["Model"].map(model_counts)

df = attach_distals(df, distals_data)
df = df.dropna(subset=FEATURE_COLS)
df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")

with open(LANG_SUBSET_PATH) as f:
    lang_list = ast.literal_eval(f.read())

df_filtered = df[
    df["Language"].str[:3].isin(lang_list) | (df["In_Data"] == 1)
].reset_index(drop=True)

df_filtered = df.reset_index(drop=True)

X      = df_filtered[FEATURE_COLS]
y      = df_filtered["In_Data"]
groups = df_filtered["Model"]

imbalance_ratio = (y == 0).sum() / (y == 1).sum()

if DRY_RUN:
    print("Dry run complete — exiting before training.")
    raise SystemExit(0)

# Hyperparameter search
if RUN_HYPERPARAM_SEARCH:
    search_results = hyperparameter_search.run_hyperparam_search(
        X, y, groups,
        n_splits=CV_SPLITS,
        rf_n_iter=RF_N_ITER,
        xgb_n_trials=XGB_N_TRIALS,
    )
    print(search_results)
    hyperparameter_search.evaluate_best(
        search_results, X, y, groups, n_splits=CV_SPLITS
    )

# Classifiers
classifiers = {
    "XGBoost": XGBClassifier(
        scale_pos_weight=imbalance_ratio,
        colsample_bylevel=0.891,
        colsample_bytree=0.964,
        gamma=0.418639,
        learning_rate=0.0651,
        max_depth=8,
        min_child_weight=6,
        n_estimators=475,
        reg_alpha=0.201408,
        reg_lambda=0.05455,
        subsample=0.9304307323537188,
    ),
}

gkf = GroupKFold(n_splits=CV_SPLITS)


def evaluate_classifier(clf_name: str, clf, X, y, groups) -> None:
    """Run GroupKFold CV for one classifier, print metrics, and save results."""
    print(f"\n{'='*40}\n  {clf_name}\n{'='*40}")

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]

        clf.fit(X_train, y_train)
        train_probs = clf.predict_proba(X_train)[:, 1]

        if USE_FBETA_THRESHOLD:
            precisions, recalls, thresholds = precision_recall_curve(y_train, train_probs)
            fbeta = (
                (1 + FBETA_BETA**2)
                * (precisions * recalls)
                / ((FBETA_BETA**2 * precisions) + recalls + 1e-8)
            )
            best_thresh = (
                thresholds[np.argmax(fbeta)]
                if np.argmax(fbeta) < len(thresholds)
                else thresholds[-1]
            )
        else:
            best_thresh = THRESHOLD

        test_probs = clf.predict_proba(X_test)[:, 1]

        train_logloss = log_loss(y_train, train_probs)
        test_logloss  = log_loss(y_test,  test_probs)
        avg_precision = average_precision_score(y_test, test_probs)

        preds    = (test_probs >= best_thresh).astype(int)
        fold_f1  = f1_score(y_test, preds)

        print(f"  Fold {fold+1} | Threshold: {best_thresh:.3f} | Test F1: {fold_f1:.3f}")
        print(f"  Fold {fold+1} | Train log-loss: {train_logloss:.4f} | Test log-loss: {test_logloss:.4f}")
        report = classification_report(y_test, preds, output_dict=True)
        print(f"  Class 1 precision: {report['1']['precision']:.4f} | recall: {report['1']['recall']:.4f}")

        fold_df = X_test.copy()
        fold_df["Model"]               = groups.iloc[test_idx].values
        fold_df["Language"]            = df_filtered.loc[test_idx, "Language"].values
        fold_df["Model_In_Data_Count"] = df_filtered.loc[test_idx, "Model_In_Data_Count"].values
        fold_df["True_Label"]          = y_test.values
        fold_df["Prediction"]          = preds
        fold_df["Probability"]         = test_probs
        fold_results.append(fold_df)

    # Aggregate and save
    df_results = (
        pd.concat(fold_results, ignore_index=True)
        .merge(size_df[["Model", "Model_Type", "Model_Size"]], on="Model", how="left")
    )
    df_out = attach_distals(df_results, distals_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"with_langs_no_meta_{clf_name}.xlsx"
    df_results.to_excel(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    y_true, y_pred = df_results["True_Label"], df_results["Prediction"]
    print(f"\n  Overall Results ({clf_name}):")
    print(f"  Precision: {precision_score(y_true, y_pred):.5f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.5f}")
    print(f"  F1:        {f1_score(y_true, y_pred):.5f}")
    print(classification_report(y_true, y_pred))

    # SHAP on full data
    clf.fit(X, y)
    explainer = shap.TreeExplainer(clf)
    rename_dict = {
        "Avg_Tok_Length": "Avg. Token Length",
        "whitespace":     "Whitespace",
        "punctuation":    "Punctuation",
        "Avg_Tok_Ratio":  "Avg. Token Ratio",
        "Unk_Ratio":      "Unk. Token Ratio",
        "Model_Type":     "Model Type",
    }
    X.rename(columns=rename_dict, inplace=True)


for clf_name, clf in classifiers.items():
    evaluate_classifier(clf_name, clf, X, y, groups)