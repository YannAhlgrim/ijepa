#!/usr/bin/env bash
#
# End-to-end script to train vith14_224_in22k on 5 seeds and produce a WILDS
# leaderboard submission tarball.
#
# This submits the 5-seed supervised training sweep via SLURM/submitit, then
# runs the submission generator to package predictions and compute F1-Macro.
#
# Usage:
#   bash tools/run_wilds_submission.sh --partition <slurm_partition> --time <minutes>
#
# The submission tarball will be written to the repo root as:
#   vith14_224_in22k_wilds_submission.tar.gz
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_NAME="vith14_224_in22k"
PARTITION=""
TIME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --time)      TIME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${PARTITION}" ]]; then
  echo "Error: --partition is required." >&2
  exit 1
fi

if [[ -z "${TIME}" ]]; then
  echo "Error: --time is required (minutes)." >&2
  exit 1
fi

echo "=============================================================="
echo "Training ${SUBMISSION_NAME} on 5 seeds via SLURM"
echo "=============================================================="

bash "${PROJECT_ROOT}/tools/run_seed_sweep.sh" \
  --partition "${PARTITION}" \
  --time "${TIME}" \
  --models "${SUBMISSION_NAME}"

echo ""
echo "=============================================================="
echo "Training jobs submitted."
echo "=============================================================="
echo ""
echo "WAIT for all SLURM jobs to finish, then generate the submission with:"
echo ""
echo "  python3 ${PROJECT_ROOT}/tools/generate_wilds_submission.py \\"
echo "    --submission-name ${SUBMISSION_NAME} \\"
echo "    --eval-root ${PROJECT_ROOT}/experiment_logs/eval-wilds \\"
echo "    --out-dir ${PROJECT_ROOT}"
echo ""
echo "This will create:"
echo "  ${PROJECT_ROOT}/${SUBMISSION_NAME}_wilds_submission.tar.gz"
echo "  ${PROJECT_ROOT}/${SUBMISSION_NAME}_seed{0..4}.pth.tar"
echo "  ${PROJECT_ROOT}/${SUBMISSION_NAME}_submission/f1_macro.txt"
