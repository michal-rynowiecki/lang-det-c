"""
How to use:

# Full training run (no hyperparameter search):
    python run.py

# With hyperparameter search:
    python run.py --hyperparam-search

# Override default CV splits and search iterations:
    python run.py --hyperparam-search --cv-splits 8 --rf-n-iter 300 --xgb-n-trials 200
"""

import argparse
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train language-inclusion classifier (optionally with hyperparameter search).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--hyperparam-search",
        action="store_true",
        default=False,
        help="Run Optuna / RandomizedSearch hyperparameter sweep before training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Load and prepare data, then exit without training or searching.",
    )

    search_group = parser.add_argument_group("hyperparameter search options")
    search_group.add_argument(
        "--rf-n-iter",
        type=int,
        default=200,
        metavar="N",
        help="Number of RandomizedSearch iterations for RandomForest.",
    )
    search_group.add_argument(
        "--xgb-n-trials",
        type=int,
        default=150,
        metavar="N",
        help="Number of Optuna trials for XGBoost.",
    )

    parser.add_argument(
        "--cv-splits",
        type=int,
        default=6,
        metavar="K",
        help="Number of GroupKFold cross-validation splits.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        metavar="T",
        help="Fixed decision threshold applied at inference (0 < T < 1).",
    )
    parser.add_argument(
        "--fbeta-beta",
        type=float,
        default=1.5,
        metavar="B",
        help="Beta value for F-beta threshold optimisation (only used when "
             "--use-fbeta-threshold is set).",
    )
    parser.add_argument(
        "--use-fbeta-threshold",
        action="store_true",
        default=False,
        help="Select per-fold threshold by maximising F-beta instead of using --threshold.",
    )

    path_group = parser.add_argument_group("path overrides")
    path_group.add_argument(
        "--distals-path",
        default="../data/distals/distals-db.pickle",
        help="Path to the distals database pickle.",
    )
    path_group.add_argument(
        "--lang-subset-path",
        default="../data/lang_subset.txt",
        help="Path to the language subset list.",
    )
    path_group.add_argument(
        "--output-dir",
        default="../data/predictions",
        help="Directory where prediction Excel files are written.",
    )

    parser.add_argument(
        "--script",
        default="../src/model.py",          # ← rename to match your actual filename
        help="Path to the main classifier script to execute.",
    )

    return parser.parse_args()


def build_env(args: argparse.Namespace):
    """
    Translate parsed arguments into environment variables that the
    classifier script reads via os.environ.
    """
    env = os.environ.copy()

    env["RUN_HYPERPARAM_SEARCH"] = "1" if args.hyperparam_search else "0"
    env["DRY_RUN"]               = "1" if args.dry_run            else "0"
    env["USE_FBETA_THRESHOLD"]   = "1" if args.use_fbeta_threshold else "0"

    env["CV_SPLITS"]     = str(args.cv_splits)
    env["THRESHOLD"]     = str(args.threshold)
    env["FBETA_BETA"]    = str(args.fbeta_beta)
    env["RF_N_ITER"]     = str(args.rf_n_iter)
    env["XGB_N_TRIALS"]  = str(args.xgb_n_trials)

    env["DISTALS_PATH"]    = args.distals_path
    env["LANG_SUBSET_PATH"] = args.lang_subset_path
    env["OUTPUT_DIR"]      = args.output_dir

    return env


def main():
    args = parse_args()

    env = build_env(args)

    result = subprocess.run(
        [sys.executable, args.script],
        env=env,
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()