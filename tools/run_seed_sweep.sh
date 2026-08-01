#!/usr/bin/env bash
#
# Run each supervised model through all seeds, one model at a time.
#
# For every model grid under configs/grids/seeds/, this submits one submitit
# job per seed (via tools/run_grid.py). Models are launched sequentially so you
# can run them "one by one"; within a model, the 5 seeds are submitted together.
#
# Each run automatically:
#   - seeds training from meta.seed (config-driven, see src/train_supervised.py)
#   - evaluates on id_test (ID) and test (OOD) WILDS splits
#   - records WILDS metrics + training time + epochs_run into the metrics JSON
#
# After all jobs finish, aggregate with:
#   python3 tools/aggregate_seeds.py --root experiment_logs/eval-wilds
#
# Usage:
#   bash tools/run_seed_sweep.sh [--partition P] [--time MIN] [--folder DIR]
#                                [--models "vith14_224 vith16_448_in1k"]
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_DIR="${PROJECT_ROOT}/configs/grids/seeds"

PARTITION=""
TIME=""
FOLDER=""
MODELS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --time)      TIME="$2"; shift 2 ;;
    --folder)    FOLDER="$2"; shift 2 ;;
    --models)    MODELS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Resolve the list of grid files to run.
if [[ -n "${MODELS}" ]]; then
  GRIDS=()
  for m in ${MODELS}; do
    g="${GRID_DIR}/${m}.yaml"
    if [[ ! -f "${g}" ]]; then
      echo "Grid not found for model '${m}': ${g}" >&2
      exit 1
    fi
    GRIDS+=("${g}")
  done
else
  # All models, sorted.
  GRIDS=()
  while IFS= read -r g; do GRIDS+=("${g}"); done < <(ls "${GRID_DIR}"/*.yaml | sort)
fi

echo "Launching seed sweeps for ${#GRIDS[@]} model(s):"
for g in "${GRIDS[@]}"; do echo "  - $(basename "${g}")"; done
echo

for g in "${GRIDS[@]}"; do
  echo "=================================================================="
  echo "Model grid: $(basename "${g}")"
  echo "=================================================================="

  cmd=("${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/tools/run_grid.py" --grid "${g}")
  [[ -n "${PARTITION}" ]] && cmd+=(--partition "${PARTITION}")
  [[ -n "${TIME}" ]]      && cmd+=(--time "${TIME}")
  [[ -n "${FOLDER}" ]]    && cmd+=(--folder "${FOLDER}")

  echo "+ ${cmd[*]}"
  "${cmd[@]}"
  echo
done

echo "All seed-sweep jobs submitted."
echo "When they finish, aggregate results with:"
echo "  ${PROJECT_ROOT}/.venv/bin/python tools/aggregate_seeds.py --root experiment_logs/eval-wilds"
