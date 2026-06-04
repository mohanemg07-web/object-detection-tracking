---
title: VisDrone Detection And Tracking
emoji: 🚁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# VisDrone Object Detection & Tracking (CPU demo)

Gradio demo for **HuggingFace Spaces** (free CPU tier: 2 vCPU / 16 GB).

Runs the fine-tuned **YOLOv8m → ONNX INT8** model via ONNX Runtime
(`CPUExecutionProvider`) with a dependency-light **ByteTrack** tracker.
Upload an aerial image for detection, or a short video for multi-object
tracking with a live FPS readout. Ten VisDrone classes: pedestrian,
people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

> The demo never uses TensorRT (GPU-only). It loads the ONNX INT8 model.

## Deploying to Spaces

The Space needs these three files at its repo root:

- `app.py`           (this folder's app)
- `requirements.txt` (this folder's CPU-only deps)
- `best.int8.onnx`   (the quantized model — add via Git LFS or download at startup)

Because `app.py` imports the `src/` package, deploy with **either**:

1. **Whole-repo Space** (simplest): push the full project to the Space and
   set the Space's `app_file` to `app/app.py`, or
2. **Standalone**: copy `app/app.py` + `app/requirements.txt` and the
   needed `src/` modules (`data/visdrone.py`, `inference/`, `tracking/`,
   `preprocessing/`) to the Space root.

Set the model location with the `MODEL_PATH` env var if it isn't at
`weights/best.int8.onnx`. Other env vars: `IMGSZ`, `CONF`,
`MAX_VIDEO_FRAMES`.

## Run locally

```bash
pip install -r app/requirements.txt
MODEL_PATH=weights/best.int8.onnx python app/app.py
```

Performance target on the free CPU tier: **~3–4 FPS**. Long videos are
capped (default 300 frames) to stay within memory/time limits.
