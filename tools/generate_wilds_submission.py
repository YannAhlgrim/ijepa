#!/usr/bin/env python3
"""
Generate a WILDS leaderboard submission tarball from multi-seed eval runs.

This script expects the post-training eval folders produced by
src/train_supervised.py, which now evaluate all four official splits and save
raw predictions as CSV files. It collects those predictions into the exact
WILDS submission naming convention, computes the per-seed and aggregated
F1-Macro, and packages everything into a .tar.gz ready for upload.

Usage:
    python3 tools/generate_wilds_submission.py \
        --submission-name vith14_224_in22k \
        --eval-root experiment_logs/eval-wilds \
        --out-dir .

Output:
    vith14_224_in22k_submission/
      iwildcam/
        iwildcam_split:val_seed:0_epoch:best_pred.csv
        iwildcam_split:id_val_seed:0_epoch:best_pred.csv
        ...
      f1_macro.txt
      summary.json
    vith14_224_in22k_wilds_submission.tar.gz
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import sys
import tarfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SPLITS = ["val", "id_val", "id_test", "test"]
PRIMARY_METRIC = "F1-macro_all"
ACC_METRIC = "acc_avg"

_SEED_SUFFIX_RE = re.compile(r"-seed(\d+)$")


def _find_metric(metrics, key):
    """Recursively search a nested dict/list for `key`."""
    if isinstance(metrics, dict):
        if key in metrics:
            return metrics[key]
        for value in metrics.values():
            found = _find_metric(value, key)
            if found is not None:
                return found
    elif isinstance(metrics, list):
        for item in metrics:
            found = _find_metric(item, key)
            if found is not None:
                return found
    return None


def _seed_from_name(run_name):
    """Parse trailing -seedN from a run folder name."""
    m = _SEED_SUFFIX_RE.search(run_name)
    return int(m.group(1)) if m else None


def _mean_std(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    else:
        std = 0.0
    return mean, std, n


def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _collect_seed_runs(eval_root, submission_name):
    """Find all run folders matching submission_name and group by seed."""
    pattern = os.path.join(eval_root, f"{submission_name}*")
    candidates = glob.glob(pattern)

    runs = []
    for path in candidates:
        if not os.path.isdir(path):
            continue
        run_name = os.path.basename(path)
        seed = _seed_from_name(run_name)
        if seed is None:
            continue
        runs.append({"run_name": run_name, "seed": seed, "path": path})

    if not runs:
        raise FileNotFoundError(
            f"No eval runs found under {eval_root} matching '{submission_name}*'"
        )

    runs.sort(key=lambda r: r["seed"])
    return runs


def _validate_run(run):
    """Check that a run folder has predictions and metrics for all splits."""
    missing = []
    files = {}
    for split in SPLITS:
        tag = f"iwildcam_{split}"
        pred_file = os.path.join(run["path"], f"{tag}_predictions.csv")
        metrics_file = os.path.join(run["path"], f"{tag}_metrics.json")
        if not os.path.exists(pred_file):
            missing.append(pred_file)
        if not os.path.exists(metrics_file):
            missing.append(metrics_file)
        files[split] = {"predictions": pred_file, "metrics": metrics_file}

    if missing:
        raise FileNotFoundError(
            f"Run {run['run_name']} is missing required files:\n" + "\n".join(missing)
        )

    return files


def _write_wilds_csv(src_pred_path, dst_path):
    """Copy predictions into the exact WILDS CSV format (one label per line)."""
    shutil.copy(src_pred_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-name",
        required=True,
        help="Submission/model name used as the run-name prefix and output file prefix.",
    )
    parser.add_argument(
        "--eval-root",
        default="experiment_logs/eval-wilds",
        help="Root folder holding per-seed eval subfolders (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory where the submission folder and tarball are written (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-folder",
        action="store_true",
        help="Keep the uncompressed submission folder after creating the tarball.",
    )
    args = parser.parse_args()

    eval_root = args.eval_root
    submission_name = args.submission_name
    out_dir = os.path.abspath(args.out_dir)

    runs = _collect_seed_runs(eval_root, submission_name)
    print(f"Found {len(runs)} seed run(s) for '{submission_name}':")
    for r in runs:
        print(f"  seed {r['seed']:>2}: {r['run_name']}")

    # Prepare submission folder layout.
    submission_dir = os.path.join(out_dir, f"{submission_name}_submission")
    iwildcam_dir = os.path.join(submission_dir, "iwildcam")
    os.makedirs(iwildcam_dir, exist_ok=True)

    # Process each seed and split.
    per_seed = []
    for run in runs:
        files = _validate_run(run)
        seed_record = {"seed": run["seed"], "run_name": run["run_name"], "splits": {}}

        for split in SPLITS:
            # WILDS expects: {dataset}_split:{split}_seed:{seed}_epoch:{epoch}_pred.csv
            dst_fname = f"iwildcam_split:{split}_seed:{run['seed']}_epoch:best_pred.csv"
            dst_path = os.path.join(iwildcam_dir, dst_fname)
            _write_wilds_csv(files[split]["predictions"], dst_path)

            metrics = _load_json(files[split]["metrics"])
            f1 = _find_metric(metrics, PRIMARY_METRIC)
            acc = _find_metric(metrics, ACC_METRIC)
            seed_record["splits"][split] = {
                "f1_macro": f1,
                "acc_avg": acc,
                "predictions_file": dst_fname,
            }

        per_seed.append(seed_record)

    # Aggregate mean +/- std across seeds for headline splits.
    summary = {
        "submission_name": submission_name,
        "num_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "primary_metric": PRIMARY_METRIC,
        "acc_metric": ACC_METRIC,
        "per_seed": per_seed,
    }

    leaderboard = {}
    for split in SPLITS:
        f1_values = [s["splits"][split]["f1_macro"] for s in per_seed]
        acc_values = [s["splits"][split]["acc_avg"] for s in per_seed]
        f1_mean, f1_std, f1_n = _mean_std(f1_values)
        acc_mean, acc_std, acc_n = _mean_std(acc_values)
        leaderboard[split] = {
            "f1_macro_mean": f1_mean,
            "f1_macro_std": f1_std,
            "f1_macro_n": f1_n,
            "acc_avg_mean": acc_mean,
            "acc_avg_std": acc_std,
            "acc_avg_n": acc_n,
        }
    summary["leaderboard"] = leaderboard

    # Generalization gap on the primary metric.
    id_test_f1 = leaderboard.get("id_test", {}).get("f1_macro_mean")
    test_f1 = leaderboard.get("test", {}).get("f1_macro_mean")
    if id_test_f1 is not None and test_f1 is not None:
        summary["generalization_gap_f1_macro"] = id_test_f1 - test_f1

    # Write summary JSON.
    summary_path = os.path.join(submission_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Write human-readable F1 macro file.
    f1_path = os.path.join(submission_dir, "f1_macro.txt")
    with open(f1_path, "w") as f:
        f.write(f"WILDS iWildCam submission: {submission_name}\n")
        f.write(f"Seeds: {summary['seeds']}\n")
        f.write(f"Primary metric: {PRIMARY_METRIC}\n\n")

        f.write("Per-seed results:\n")
        header = f"{'seed':>6}  {'val':>10}  {'id_val':>10}  {'id_test':>10}  {'test':>10}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for s in per_seed:
            f.write(
                f"{s['seed']:>6}  "
                f"{s['splits']['val']['f1_macro'] or 0.0:>10.4f}  "
                f"{s['splits']['id_val']['f1_macro'] or 0.0:>10.4f}  "
                f"{s['splits']['id_test']['f1_macro'] or 0.0:>10.4f}  "
                f"{s['splits']['test']['f1_macro'] or 0.0:>10.4f}\n"
            )

        f.write("\nAggregated (mean +/- std):\n")
        for split in SPLITS:
            entry = leaderboard[split]
            f.write(
                f"{split:>10}: F1-Macro = {entry['f1_macro_mean']:.4f} "
                f"+/- {entry['f1_macro_std']:.4f}\n"
            )
            f.write(
                f"{'':>10}  Acc-Avg  = {entry['acc_avg_mean']:.4f} "
                f"+/- {entry['acc_avg_std']:.4f}\n"
            )

        gap = summary.get("generalization_gap_f1_macro")
        if gap is not None:
            f.write(f"\nID -> OOD F1-Macro gap (id_test - test): {gap:.4f}\n")

    # Create tarball.
    tarball_name = f"{submission_name}_wilds_submission.tar.gz"
    tarball_path = os.path.join(out_dir, tarball_name)
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(submission_dir, arcname=os.path.basename(submission_dir))

    print(f"\nSubmission folder: {submission_dir}")
    print(f"Upload tarball:    {tarball_path}")
    print(f"F1-Macro summary:  {f1_path}")
    print("\nHeadline numbers:")
    for split in SPLITS:
        entry = leaderboard[split]
        print(
            f"  {split:>10}: F1-Macro = {entry['f1_macro_mean']:.4f} "
            f"+/- {entry['f1_macro_std']:.4f}"
        )
    if gap is not None:
        print(f"  ID -> OOD gap:   {gap:.4f}")

    if not args.keep_folder:
        print(f"\nCleaning up uncompressed folder {submission_dir}")
        shutil.rmtree(submission_dir)


if __name__ == "__main__":
    main()
