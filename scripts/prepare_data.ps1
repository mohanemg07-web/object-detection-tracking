# Download + convert + validate the VisDrone dataset (Windows).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/prepare_data.ps1
$ErrorActionPreference = "Stop"

python -m src.data.download_visdrone --config configs/paths.yaml
python -m src.data.convert_visdrone  --config configs/paths.yaml
python -m src.data.validate_labels   --config configs/paths.yaml
