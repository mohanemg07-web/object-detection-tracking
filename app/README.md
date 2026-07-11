---
title: VisDrone Detection And Tracking
emoji: 🚁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.19.2
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# VisDrone Object Detection & Tracking (CPU demo)

Gradio demo for **HuggingFace Spaces** (free CPU tier: 2 vCPU / 16 GB).

▶️ **Live:** [huggingface.co/spaces/MohanGen/visdrone-detection-tracking](https://huggingface.co/spaces/MohanGen/visdrone-detection-tracking)

Runs the fine-tuned **YOLOv8m → FP32 ONNX** model via ONNX Runtime
(`CPUExecutionProvider`) with a dependency-light **ByteTrack** tracker.
Upload an aerial image for detection, or a short video for multi-object
tracking with a live FPS readout. Ten VisDrone classes: pedestrian,
people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

> The demo runs the **FP32 ONNX** model, **not** INT8. The INT8 model passed
> tensor-level parity (max abs diff 1.1e-3) but produces **zero detections**
> on real images (all class scores ≈ 0) — real-image validation caught what
> the near-zero tensor diff missed. The demo also never uses TensorRT
> (GPU-only).

Measured on the free CPU tier: a single image with **67 objects in 1514 ms**;
a **192-frame** video tracked at **avg 0.8 FPS** (ONNX Runtime CPU + ByteTrack).

## Deploying to Spaces

The Space needs these three files at its repo root:

- `app.py`           (this folder's app)
- `requirements.txt` (this folder's CPU-only deps)
- `best.onnx`        (the FP32 ONNX model — add via Git LFS or download at startup; **not** the INT8 model, which produces zero detections)

Because `app.py` imports the `src/` package, deploy with **either**:

1. **Whole-repo Space** (simplest): push the full project to the Space and
   set the Space's `app_file` to `app/app.py`, or
2. **Standalone**: copy `app/app.py` + `app/requirements.txt` and the
   needed `src/` modules (`data/visdrone.py`, `inference/`, `tracking/`,
   `preprocessing/`) to the Space root.

Set the model location with the `MODEL_PATH` env var if it isn't at
`weights/best.onnx`. Other env vars: `IMGSZ`, `CONF`, `MAX_VIDEO_FRAMES`.

## Run locally

```bash
pip install -r app/requirements.txt
MODEL_PATH=weights/best.onnx python app/app.py
```

Measured on the free CPU tier: single image with 67 objects in **1514 ms**;
a 192-frame video tracked at **avg 0.8 FPS**. Long videos are capped
(default 300 frames) to stay within memory/time limits.
