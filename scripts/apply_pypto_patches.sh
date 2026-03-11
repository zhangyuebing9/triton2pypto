#!/bin/bash
# Apply pypto compatibility patches for simpler a2a3sim.
# Run from repo root after: git submodule update --init third_party/pypto

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYPTO_DIR="$REPO_ROOT/third_party/pypto"
PATCH_FILE="$REPO_ROOT/patches/pypto-simpler-compat.patch"

if [[ ! -d "$PYPTO_DIR" ]]; then
    echo "Error: third_party/pypto not found. Run: git submodule update --init third_party/pypto"
    exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
    echo "Error: Patch file not found: $PATCH_FILE"
    exit 1
fi

cd "$PYPTO_DIR"
if patch -p1 --forward --dry-run < "$PATCH_FILE" 2>/dev/null; then
    patch -p1 --forward < "$PATCH_FILE"
    echo "Applied pypto simpler compatibility patch."
else
    echo "Patch already applied or not applicable."
fi
