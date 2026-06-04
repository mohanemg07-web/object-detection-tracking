#!/usr/bin/env bash
# POSIX setup: create a virtual environment and install CPU/base deps.
# Usage:  bash scripts/setup.sh
set -euo pipefail

echo "Creating virtual environment (.venv) ..."
python3 -m venv .venv

echo "Activating and upgrading pip ..."
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "Installing CPU/base requirements ..."
pip install -r requirements.txt

echo "Done. Activate later with:  source .venv/bin/activate"
