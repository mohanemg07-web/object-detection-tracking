# Convenience entrypoints. On Windows, run the equivalent commands
# directly or use `scripts/*.ps1`. These targets assume a POSIX shell.

.PHONY: help install install-gpu lint format test \
        data-download data-convert data-validate data \
        train export quantize benchmark track demo clean

help:
	@echo "Targets:"
	@echo "  install        Install CPU/base deps"
	@echo "  install-gpu    Install GPU deps (training + TensorRT)"
	@echo "  lint           Run ruff + black --check"
	@echo "  format         Auto-format with black + ruff --fix"
	@echo "  test           Run pytest"
	@echo "  data           Download + convert + validate VisDrone"
	@echo "  train          Fine-tune YOLOv8m (GPU)"
	@echo "  export         Export best.pt -> ONNX"
	@echo "  quantize       ONNX INT8 static quantization"
	@echo "  benchmark      FP32 vs ONNX-INT8 vs TensorRT-INT8"
	@echo "  track          Run unified detect+track inference"
	@echo "  demo           Launch the Gradio app locally (CPU)"

install:
	pip install -r requirements.txt

install-gpu:
	pip install -r requirements.txt
	pip install -r requirements-gpu.txt

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

test:
	pytest

data-download:
	python -m src.data.download_visdrone --config configs/paths.yaml

data-convert:
	python -m src.data.convert_visdrone --config configs/paths.yaml --write-data-yaml

data-validate:
	python -m src.data.validate_labels --config configs/paths.yaml

data: data-download data-convert data-validate

train:
	python -m src.training.train --config configs/train.yaml

export:
	python -m src.optimization.export_onnx --weights weights/best.pt

quantize:
	python -m src.optimization.quantize_int8 --weights weights/best.onnx

benchmark:
	python -m src.optimization.benchmark

track:
	python -m src.inference.run --source 0

demo:
	python app/app.py

clean:
	rm -rf runs outputs __pycache__ .pytest_cache .ruff_cache
