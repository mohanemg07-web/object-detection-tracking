# Windows setup: create a virtual environment and install CPU/base deps.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment (.venv) ..."
python -m venv .venv

Write-Host "Activating and upgrading pip ..."
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Write-Host "Installing CPU/base requirements ..."
pip install -r requirements.txt

Write-Host "Done. Activate later with:  .\.venv\Scripts\Activate.ps1"
