#!/bin/bash
# Install the Triton wheel supplied by the trimmed build pipeline.
set -euo pipefail

PYTHON="python3"
PIP="$PYTHON -m pip"

if [[ -z "${TRITON_WHEEL:-}" ]]; then
    echo "TRITON_WHEEL must point to the organization's Triton wheel" >&2
    exit 2
fi

${PIP} install --force-reinstall --no-deps "${TRITON_WHEEL}"
