#!/usr/bin/env bash
set -euo pipefail

# Resolve the entry point independently of the caller's working directory.
# Clear CDPATH so that a shell setting cannot alter or print this directory.
CODE_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults are repository-relative. Explicit relative paths (including PYTHON)
# remain relative to the caller: do not change the working directory here.
# Arguments supplied after these defaults take precedence in run.py.
exec "${PYTHON:-python}" -u "$CODE_DIR/run.py" \
  --data-root "${DATA_ROOT:-$CODE_DIR/dataset}" \
  --output-dir "${OUTPUT_DIR:-$CODE_DIR/results/paper}" "$@"
