#!/usr/bin/env bash
# UGSLoc — Cambridge Landmarks localization (paper-style defaults are built into the Python entry).

set -euo pipefail

SCENE_ROOT="${SCENE_ROOT:-/path/to/scene}"
scenes=(KingsCollege ShopFacade OldHospital StMarysChurch)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}/ugsLoc"

for scene in "${scenes[@]}"; do
  echo "=== ${scene} ==="
  python loc.py --dataset cambridge \
    -s "${SCENE_ROOT}/${scene}" \
    -m "${SCENE_ROOT}/${scene}" \
    --scene_name "${scene}"
done

echo "Done."
