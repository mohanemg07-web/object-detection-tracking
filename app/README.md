# app/

Gradio demo for **HuggingFace Spaces** (free CPU tier: 2 vCPU / 16 GB).

Runs the **ONNX INT8** detection model via ONNX Runtime
(`CPUExecutionProvider`) with ByteTrack, drawing detections + tracks and
reporting FPS. Dependency-light by design — see `app/requirements.txt`
(added in Phase 6).

> The demo never uses TensorRT (GPU-only). It loads `weights/best.int8.onnx`.
