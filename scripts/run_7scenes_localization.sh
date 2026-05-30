#!/usr/bin/env bash
# UGSLoc — 7-Scenes localization (paper-style defaults are built into the Python entry).

set -euo pipefail

SCENE_ROOT="${SCENE_ROOT:-/path/to/scene}"
GAUSSIAN_ROOT="${GAUSSIAN_ROOT:-/path/to/gaussian_model}"
scenes=(chess fire heads office pumpkin redkitchen stairs)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}/ugsLoc"

for scene in "${scenes[@]}"; do
  echo "=== ${scene} ==="
  python loc.py --dataset 7scenes \
    -s "${SCENE_ROOT}/pgt_7scenes_${scene}/test" \
    -m "${GAUSSIAN_ROOT}/scene_${scene}/train/output" \
    --scene_name "${scene}"
done

echo "Done."
