"""Gradio demo (HuggingFace Spaces, CPU) — VisDrone detection + tracking.

Thin UI layer over ``app/demo_logic.py``. All inference runs on the ONNX
INT8 model via ONNX Runtime (CPUExecutionProvider) with a dependency-light
ByteTracker — no torch, no TensorRT. Designed for the free Spaces tier
(2 vCPU / 16 GB).

Run locally:
    python app/app.py
On Spaces this file is the entrypoint (see app/README.md header).
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

# Allow `python app/app.py` and Spaces (cwd=app/) to find demo_logic + src.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from demo_logic import (  # noqa: E402
    CLASS_NAMES,
    MAX_VIDEO_FRAMES,
    detect_image,
    detect_video,
    missing_model_message,
    model_available,
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="VisDrone Detection & Tracking (CPU)") as demo:
        gr.Markdown(
            "# VisDrone Object Detection & Tracking\n"
            "YOLOv8m (ONNX **INT8**) + ByteTrack, running on **CPU** via ONNX "
            "Runtime. Upload an aerial image for detection, or a short video "
            "for multi-object tracking. 10 classes: "
            f"{', '.join(CLASS_NAMES)}."
        )
        if not model_available():
            gr.Markdown(f"> ⚠️ {missing_model_message()}")

        with gr.Tab("Image"):
            with gr.Row():
                img_in = gr.Image(type="numpy", label="Input image")
                img_out = gr.Image(type="numpy", label="Detections")
            img_info = gr.Textbox(label="Summary", lines=2)
            gr.Button("Detect").click(detect_image, img_in, [img_out, img_info])

        with gr.Tab("Video"):
            vid_in = gr.Video(label="Input video")
            vid_out = gr.Video(label="Tracked output")
            vid_info = gr.Textbox(label="Summary", lines=2)
            gr.Button("Track").click(detect_video, vid_in, [vid_out, vid_info])

        gr.Markdown(
            "_Free CPU tier (2 vCPU / 16 GB). Target ~3–4 FPS; long videos are "
            f"capped at {MAX_VIDEO_FRAMES} frames._"
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
