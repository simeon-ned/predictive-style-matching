#!/usr/bin/env bash
set -euo pipefail

# Pull all LFS blobs under data/ on demand.
# Run from anywhere inside the repo:
#   bash src/psm/scripts/lfs_pull_data.sh

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Fetching LFS objects for data/** ..."
# Use explicit fetch+checkout so it works even when .lfsconfig sets fetchexclude=data/**.
git lfs fetch --include="data/**" --exclude=""

echo "Checking out pulled files in working tree ..."
git lfs checkout -- "data"

echo "Done. LFS data is available under: $REPO_ROOT/data"
